from keras.models import model_from_json
import cv2
import numpy as np

# Load model architecture
json_file = open("signLanguageInterpreter.json", "r")
model_json = json_file.read()
json_file.close()

# Load model weight
model = model_from_json(model_json)
model.load_weights("signLanguageInterpreter.keras")

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 50, 50, 1)
    return feature / 255.0

cap = cv2.VideoCapture(0)

# labels
label = ['A', 'B', 'C', 'D', 'E', 'F', 'ack','space']

sentence = ""
prev_label = "blank"
confirmed_letter = None
frame_counter = 0
confidence_threshold = 0.7
required_frames = 15

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    # ROI region
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cropframe = frame[40:300, 0:300]

    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (50, 50))

    # extract features and predict
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_confidence = pred.max()
    prediction_label = label[pred.argmax()]

    if prediction_confidence >= confidence_threshold:
        if prediction_label == prev_label and prediction_label != "ack":
            frame_counter += 1
        else:
            frame_counter = 0

        if frame_counter >= required_frames:
            confirmed_letter = prediction_label
            frame_counter = 0

        if confirmed_letter and prediction_label == "ack":
            if confirmed_letter != 'space':
                sentence += confirmed_letter
            else:
                sentence += ''
            confirmed_letter = None

        previous_label = prediction_label

    cv2.putText(frame, f'Current: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)


    # sentence block
    height, width, _ = frame.shape
    print(height, width)
    cv2.rectangle(frame, (0, height - 50), (width, height), (0, 0, 0), -1)

    cv2.putText(frame, 'Sentence: %s' % (sentence), (10, height - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    cv2.imshow("output", frame)

    if cv2.waitKey(45) == ord('#'):
        break
cap.release()
cv2.destroyAllWindows()
