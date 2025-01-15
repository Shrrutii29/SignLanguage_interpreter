import cv2
import numpy as np
import textwrap
import torch
import torch.nn.functional as F
from torchvision import transforms

# Load the trained PyTorch model
class SignLanguageModel(torch.nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        # Define your model architecture here if not pre-saved
        pass

model = SignLanguageModel()
model.load_state_dict(torch.load("signLanguageInterpreter.pth"))
model.eval()

# Preprocess input image
def extract_features(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# Labels for predictions
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'blank', 'space']

# Initialization
sentence = ""
prev_label = None
cnt = 0

# Accuracy computation
correct_predictions = 0
total_predictions = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Region of interest (ROI)
    cv2.rectangle(frame, (40, 40), (300, 300), (255, 255, 255), 2)
    roi = frame[40:300, 0:300]

    # Extract features and predict
    roi_features = extract_features(roi)
    with torch.no_grad():
        pred = model(roi_features)
        probabilities = F.softmax(pred, dim=1)
        max_prob, pred_idx = torch.max(probabilities, dim=1)
        prediction_label = labels[pred_idx.item()]

    # Confirm gesture if prediction is consistent for 25 frames
    if prediction_label == prev_label:
        cnt += 1
    else:
        cnt = 0

    if cnt >= 25:
        if prediction_label == 'space':
            sentence += ' '
        elif prediction_label == 'blank':
            pass
        else:
            sentence += prediction_label
        cnt = 0

    prev_label = prediction_label

    # Ask for ground truth label for accuracy
    ground_truth_label = input("Enter ground truth label (or '#' to stop): ").strip()
    if ground_truth_label == "#":
        break

    # Update accuracy metrics
    if ground_truth_label in labels:  # Ensure valid input
        total_predictions += 1
        if prediction_label == ground_truth_label:
            correct_predictions += 1
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%")

    # Break sentence to fit on screen
    lines = textwrap.wrap(sentence, width=40)
    lines.insert(0, "Translation:")

    # Sentence block
    block_height = 20 * (len(lines) + 1)
    cv2.rectangle(frame, (0, frame.shape[0] - block_height - 10), 
                  (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        y_pos = frame.shape[0] - block_height + (20 * i) + 15
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    # Display current prediction
    cv2.putText(frame, f'Current: {prediction_label}', (10, 30), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Interpreter", frame)

    # '#' to exit
    if cv2.waitKey(1) == ord('#'):
        break

cap.release()
cv2.destroyAllWindows()

# Final accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")
else:
    print("No predictions made.")

