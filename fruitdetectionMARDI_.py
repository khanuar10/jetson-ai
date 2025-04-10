import os
import torch
import torchvision
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt

# Classes in your dataset (adjust if needed)
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Load the trained Faster R-CNN model
def load_trained_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # safer load
    model.eval()
    return model

# Prediction function
def predict(image, model, device):
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        prediction = model([image])
    return prediction

# Visualization function
def visualize_prediction(image, prediction, threshold=0.4):
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    # Filter boxes by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if boxes.nelement() == 0:
        print("No objects detected with confidence >= threshold.")
        return

    # Convert class indices to names and build text annotations
    class_names = [classes[i] for i in labels]
    text = [f"{name}: {score:.2f}" for name, score in zip(class_names, scores)]

    # Draw boxes (convert to uint8 image)
    drawn_image = draw_bounding_boxes((image.cpu() * 255).byte(), boxes, labels=text, width=4)

    # Show image
    plt.figure(figsize=(12, 8))
    plt.imshow(drawn_image.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Detections")
    plt.show()

# Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

# Device selection
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Load model
model_path = 'fruit_detect_statedict.pth'  # 🔁 Make sure this path is correct
trained_model = load_trained_model(model_path, num_classes)

# Load and preprocess image
image_path = 'fruit/Test/00b4ebd4-orange_8.jpeg'  # 🔁 Replace with a valid path
image = preprocess_image(image_path)

# Perform inference and visualize
prediction = predict(image, trained_model, device)
visualize_prediction(image, prediction)
