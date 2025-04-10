import os
import cv2
import torch
import torchvision
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np

# Classes in your dataset
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
def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    return prediction

# Frame preprocessing
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

# Draw prediction on frame
def draw_predictions(frame, prediction, threshold=0.4):
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']

    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.nelement() == 0:
        return frame  # no detections

    class_names = [classes[i] for i in labels]
    texts = [f"{name}: {score:.2f}" for name, score in zip(class_names, scores)]

    # Convert frame to tensor for drawing
    frame_tensor = transforms.ToTensor()(frame).mul(255).byte()
    annotated = draw_bounding_boxes(frame_tensor, boxes, labels=texts, width=3, font_size=16)
    return annotated.permute(1, 2, 0).cpu().numpy()

# Setup
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_path = 'fruit_detect_statedict.pth'  # 🔁 Make sure the model path is valid
trained_model = load_trained_model(model_path, num_classes)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess and predict
    frame_tensor = preprocess_frame(frame)
    prediction = predict(frame_tensor, trained_model, device)

    # Draw predictions
    annotated_frame = draw_predictions(frame, prediction, threshold=0.5)

    # Show frame
    cv2.imshow("Faster R-CNN - Real-time Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
