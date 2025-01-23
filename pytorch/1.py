import torch
import torch.nn as nn
from torchvision import transforms, datasets
import cv2
from PIL import Image
import textwrap

# Model class
class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),

            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8),  
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignLanguageCNN().to(device)
model.load_state_dict(torch.load("signLanguageInterpreter2.pth", map_location=device))

# evaluate model
model.eval()

# augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Labels for predictions
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'blank', 'space']
confidence_threshold = 0.9

# Initialization
sentence = ""
prev_label = None
cnt = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    roi = frame[40:300, 0:300]

    # Apply the preprocessing steps
    roi_pil = Image.fromarray(roi)
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        logits = model(roi_tensor)
        confidence, predicted_class = torch.max(logits, dim=1)  

    # Mapping prediction to label
    prediction_label = labels[predicted_class.item()] 
    if confidence.item() >= confidence_threshold:
        prediction_label = labels[predicted_class.item()] 
    else:
        prediction_label = "unknown"

    # Sentence building
    if prediction_label == prev_label:
        cnt += 1
    else:
        cnt = 0

    if cnt >= 25:
        if prediction_label == 'space':
            sentence += ' '
        elif prediction_label == 'blank' or prediction_label == "unknown":
            pass
        else:
            sentence += prediction_label
        cnt = 0

    prev_label = prediction_label

    # Display translation and predictions
    lines = textwrap.wrap(f"Translation: {sentence}", width=40)
    block_height = 20 * (len(lines) + 1)
    cv2.rectangle(frame, (0, frame.shape[0] - block_height - 10), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y_pos = frame.shape[0] - block_height + (20 * i) + 15
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Current: {prediction_label} ({confidence.item():.2f})", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Sign Language Interpreter", frame)

    # Exit when '#' key is pressed
    if cv2.waitKey(1) == ord('#'):
        break

cap.release()
cv2.destroyAllWindows()

