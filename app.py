"""import streamlit as st
from io import BytesIO
import numpy as np
from torchcam.methods import GradCAM
from torchvision.models import resnet50, resnet18, ResNet50_Weights , ResNet18_Weights
from torch import nn
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Define classes (if not appending to Xray.py)
classes = ['NORMAL', 'PNEUMONIA', 'COVID', 'TB']

# Define transformations (if not appending to Xray.py)
test_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])

# Define device (if not appending to Xray.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model if saved (uncomment if using separate app.py, comment out if using in-memory model)

@st.cache_resource
def load_model():
    try:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load('xray_model.pth', map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'xray_model.pth' not found. Please save the trained model using torch.save(model.state_dict(), 'xray_model.pth').")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()


# Streamlit app
st.title('X-ray Classifier')
st.write("Upload a PNG X-ray image to classify (NORMAL, PNEUMONIA, COVID, TB) and view the Grad-CAM heatmap.")

uploaded_file = st.file_uploader('Upload X-ray (PNG)', type=['png'])

if uploaded_file:
    try:
        # Read uploaded file as bytes and convert to OpenCV image
        bytes_data = uploaded_file.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error(f"Failed to load image: {uploaded_file.name}")
            st.stop()

        # Apply test transform
        img_tensor = test_transform(image=img)['image'].unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        # Get model prediction
        model.eval()
        pred = model(img_tensor)  # No torch.no_grad() here — GradCAM needs gradients
        probs = torch.softmax(pred, dim=1).detach().cpu().numpy()[0]
        #probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        class_idx = int(probs.argmax())  # Convert to Python int to avoid numpy.int64 issues
        #with torch.no_grad():
        #    pred = model(img_tensor)
        #probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        #class_idx = probs.argmax()
        st.write(f'Prediction: {classes[class_idx]} ({probs.max()*100:.1f}%)')

        # Compute Grad-CAM
        cam = GradCAM(model, target_layer='layer4')
        cam_map = cam(class_idx=class_idx, scores=model(img_tensor))[0].cpu().numpy()
        cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

        # Display images
        st.image(img, caption='Original X-ray', use_container_width=True, channels='GRAY')
        st.image(cam_map, caption=f'Grad-CAM: {classes[class_idx]}', use_container_width=True, clamp=True)

        # Overlay heatmap
        st.write("Overlay: Original X-ray with Grad-CAM")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray')
        ax.imshow(cam_map, cmap='jet', alpha=0.5)
        ax.set_title(f'Grad-CAM Overlay: {classes[class_idx]}')
        ax.axis('off')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {e}")"""



import streamlit as st
from io import BytesIO
import numpy as np
from torchcam.methods import GradCAM
from torchvision.models import resnet50, resnet18, ResNet50_Weights , ResNet18_Weights, densenet201, DenseNet201_Weights, efficientnet_b7, EfficientNet_B7_Weights
from torch import nn
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Define classes (if not appending to Xray.py)
classes = ['NORMAL', 'PNEUMONIA', 'COVID', 'TB']

# Define transformations (if not appending to Xray.py)
test_transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])

# Define device (if not appending to Xray.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_options = {
    "ResNet18": "xray_model",
    "ResNet50": "res50model",
    "DenseNet201": "dennet",
    #"EfficientNetB7": "effnetmodel"
}
selected_label = st.sidebar.selectbox("Choose model:", list(model_options.keys()))
model_key = model_options[selected_label]
#st.sidebar.title("Model Selection")
#selected_model = st.sidebar.selectbox("Choose model:", ["xray_model(ResNet18)", "res50(ResNet50)", "dennet(DenseNet201)", "effnetmodel(EfficientNetB7)"])


# Load saved model 
@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "xray_model":#(ResNet18)":#"ResNet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "res50model":#(Resnet50)":#"ResNet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "dennet":#(DenseNet201)":#"DenseNet121":
            model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        #elif model_name == "effnetmodel":#(EfficientNetB7)":#"EfficientNetB7":
        #    model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        else:
            st.error("Unsupported model selected.")
            st.stop()

        # Update first conv layer if grayscale input
        if model_name.startswith("xray_model"):
            model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3)#, bias=False)
            model.fc = nn.Linear(model.fc.in_features, 4)
        elif model_name == "res50model":
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
            model.fc = nn.Linear(model.fc.in_features, 4)
        elif model_name == "dennet":
            old_weights = model.features.conv0.weight.data
            model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride= 2, padding = 3, bias=False)
            model.features.conv0.weight.data = old_weights.mean(dim=1, keepdim=True)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(model.classifier.in_features, len(classes))
        #elif model_name == "effnetmodel":
        #    conv0 = model.features[0][0]
        #    new_conv = nn.Conv2d(
        #        in_channels=1,
        #        out_channels=conv0.out_channels,
        #        kernel_size=conv0.kernel_size,
        #        stride=conv0.stride,
        #        padding=conv0.padding,
        #        bias=conv0.bias is not None
        #    )
        #    model.features[0][0] = new_conv
        #    model.classifier[1] = nn.Linear(model.classifier, len(classes))

        model.load_state_dict(torch.load(f'{model_name.lower()}.pth', map_location=device))
        model.to(device)
        model.eval()
        return model

    except FileNotFoundError:
        st.error(f"Model file '{model_name.lower()}.pth' not found. Please save it with torch.save(model.state_dict(), '{model_name.lower()}.pth').")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

#model_key = load_model(model_name=selected_model.split('(')[0].strip())
model = load_model(model_key)

#model = load_model(model_key)

# Streamlit app
st.title('X-ray Classifier')
st.write("Upload a PNG X-ray image to classify (NORMAL, PNEUMONIA, COVID, TB) and view the Grad-CAM heatmap.")

uploaded_file = st.file_uploader('Upload X-ray (PNG, JPG, JPEG)', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        # Read uploaded file as bytes and convert to OpenCV image
        bytes_data = uploaded_file.read()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error(f"Failed to load image: {uploaded_file.name}")
            st.stop()

        # Apply test transform
        img_tensor = test_transform(image=img)['image'].unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        # Get model prediction
        model.eval()
        pred = model(img_tensor)  # No torch.no_grad() here — GradCAM needs gradients
        probs = torch.softmax(pred, dim=1).detach().cpu().numpy()[0]
        #probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        class_idx = int(probs.argmax())  # Convert to Python int to avoid numpy.int64 issues
        #with torch.no_grad():
        #    pred = model(img_tensor)
        #probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        #class_idx = probs.argmax()
        st.write(f'Prediction: {classes[class_idx]} ({probs.max()*100:.1f}%)')

        # Compute Grad-CAM
        # Dynamically choose target layer
        try:
            if model_key == "xray_model" or model_key == "res50model":
                target_layer = "layer4"
            elif model_key == "dennet":
                target_layer = "features.denseblock4"
            #elif model_key == "effnetmodel":
                #target_layer = "features.8"
            else:
                st.error("GradCAM target layer not defined for selected model.")
                st.stop()
            
            # The code below is useful up to the cam_map before except
            cam = GradCAM(model, target_layer=target_layer)
            cam_map = cam(class_idx=class_idx, scores=model(img_tensor))[0].cpu().numpy()
            cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

            # Compute Grad-CAM
            #cam = GradCAM(model, target_layer=target_layer)
            #cam_map = cam(class_idx=class_idx, scores=model(img_tensor))[0].cpu().numpy()

            # Normalize and convert CAM to 8-bit grayscale
            #cam_map = np.maximum(cam_map, 0)
            #cam_map = cam_map / (cam_map.max() + 1e-8)  # avoid divide by zero
            #cam_map_resized = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
            #cam_map_uint8 = (cam_map_resized * 255).astype(np.uint8)  # ensure type uint8 and shape [H,W]


# Normalize CAM
            #cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

# Resize CAM to original image size
            #cam_map_resized = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
            # Convert to 8-bit unsigned int
            #cam_map_uint8 = np.uint8(cam_map_resized * 255)

# Apply color map
            #heatmap = cv2.applyColorMap(cam_map_uint8, cv2.COLORMAP_JET)

# Convert grayscale image to RGB
            #img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Overlay Grad-CAM heatmap on original image
            #overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

# Add class label and probability on the image
            #label_text = f"{classes[class_idx]}: {probs[class_idx]*100:.2f}%"
            #cv2.putText(overlay, label_text, (10, overlay.shape[0] - 10),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            st.error(f"GradCAM failed: {e}")
            st.stop()
    #cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
        #cam = GradCAM(model, target_layer='layer4')
        #cam_map = cam(class_idx=class_idx, scores=model(img_tensor))[0].cpu().numpy()
        #cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
        #cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

        # Display images
        st.image(img, caption='Original X-ray', use_container_width=True, channels='GRAY')
        st.image(cam_map, caption=f'Grad-CAM: {classes[class_idx]}', use_container_width=True, clamp=True)
        #st.image(overlay, caption='Overlay: Original X-ray with Grad-CAM', use_container_width=True)

        # Convert cam_map to heatmap
        #heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Ensure cam_map is normalized to 0-255 and converted to uint8
        #cam_map_uint8 = np.uint8(255 * cam_map)
        #if len(cam_map_uint8.shape) == 3:
            #cam_map_uint8 = cam_map_uint8.squeeze()  # Remove singleton channel if present

# Now apply colormap
        #heatmap = cv2.applyColorMap(cam_map_uint8, cv2.COLORMAP_JET)
        #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


        # Convert grayscale X-ray to RGB
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Overlay heatmap on original X-ray
        #overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

# Add prediction probability text (e.g., 90.45%)
        #label_text = f"{probs[class_idx]*100:.2f}%"
        #font_scale = 1.4
        #font_thickness = 3
        #text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        #text_x = int((overlay.shape[1] - text_size[0]) / 2)
        #text_y = overlay.shape[0] - 10

# Draw black border for better contrast
        #cv2.putText(overlay, label_text, (text_x, text_y),
                    #cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
# Draw white text
        #cv2.putText(overlay, label_text, (text_x, text_y),
                    #cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Show in Streamlit
        #st.image(overlay, caption='Grad-CAM Overlay with Prediction', use_container_width=True)


        # Overlay heatmap
        st.write("Overlay: Original X-ray with Grad-CAM")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray')
        ax.imshow(cam_map, cmap='jet', alpha=0.5)
        ax.set_title(f'Grad-CAM Overlay: {classes[class_idx]}')
        ax.axis('off')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {e}")

