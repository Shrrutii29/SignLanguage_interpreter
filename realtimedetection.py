import cv2
import numpy as np
from keras.models import model_from_json
import textwrap

# Load the model architecture and weights
with open("signLanguageInterpreter.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("signLanguageInterpreter.keras")

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image).reshape(1, 50, 50, 1)
    return feature / 255.0

# Labels for predictions
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'blank', 'space']

# Initialization
sentence = ""
prev_label = None
frame_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    roi = frame[40:300, 0:300]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (50, 50))

    # Extract features and do predictions
    roi_features = extract_features(roi)
    pred = model.predict(roi_features)
    prediction_label = labels[pred.argmax()]

    # Check if the gesture is consistent for required frames
    if prediction_label == prev_label:
        frame_counter += 1
    else:
        frame_counter = 0

    # Confirm the gesture after 
    if frame_counter >= 25:
        if prediction_label == 'space':
            sentence += ' '
        elif prediction_label == 'blank':
            pass
        else:
            sentence += prediction_label

        frame_counter = 0

    prev_label = prediction_label

    # Break the sentence to fit in screen
    lines = textwrap.wrap(sentence, width = 40)

    # Sentence Label
    lines.insert(0, "Sentences : ")

    # sentence block
    block_height = 20 * (len(lines) + 1)
    cv2.rectangle(frame, (0, frame.shape[0] - block_height - 10), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

    # Display lines
    for i, line in enumerate(lines):
        y_pos = frame.shape[0] - block_height + (20 * i) + 15
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        
    # Display the current prediction
    cv2.putText(frame, f'Current: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Interpreter", frame)

    # Use '#' to exit
    if cv2.waitKey(1) == ord('#'):
        break

cap.release()
cv2.destroyAllWindows()
