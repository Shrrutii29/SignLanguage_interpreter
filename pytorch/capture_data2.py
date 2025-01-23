import os
import cv2
import mediapipe as mp

data_dir = './dataset2'
limit = 100

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

def remove_background_and_extract_hand(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1]
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1]
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0]
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0]
            
            # Crop the hand region
            hand_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            return hand_image  # Return the cropped hand image
    return None  # Return None if no hand is detected

while True:
    # Get input from user
    alphabet = input("Enter alphabet to collect images (press -1 to stop): ")
    
    # Stop execution if the user enters '-1'
    if alphabet == '-1':
        break
    
    # Create folder for respective alphabet if it doesn't exist
    alphabet_dir = os.path.join(data_dir, str(alphabet))
    if not os.path.exists(alphabet_dir):
        os.makedirs(alphabet_dir)
    
    # Count number of existing images to avoid overwriting
    existing_images = len(os.listdir(alphabet_dir))

    print(f'Collecting data for alphabet {alphabet}')
    
    # Wait till '#' is entered
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: unable to capture image")
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Display message and ROI rectangle
        cv2.putText(frame, 'Press # to start', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)

        cv2.imshow('frame', frame)
        
        # Enter '#' to start capturing images
        if cv2.waitKey(45) == ord('#'):
            break
    
    # Capture the specified number of images
    counter = existing_images
    while counter < existing_images + limit:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: unable to capture image")
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # ROI rectangle
        cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)

        # Crop the ROI
        cropframe = frame[40:300, 0:300]

        # Remove background and extract hand
        hand_image = remove_background_and_extract_hand(cropframe)

        if hand_image is not None:
            # Convert to grayscale and resize input
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            hand_image = cv2.resize(hand_image, (50, 50))

            # Save cropped hand image
            cv2.imwrite(os.path.join(alphabet_dir, '{}.jpg'.format(counter)), hand_image)
            counter += 1
            
            # Show frame
            cv2.imshow('frame', frame)
            cv2.imshow('Cropped Hand Image', hand_image)
        else:
            print(f"No hand detected in the captured frame.")

        cv2.waitKey(40)
    
    # Acknowledge
    print(f"Done collecting data for {alphabet}")

# Release camera
camera.release()

# End current session
cv2.destroyAllWindows()
