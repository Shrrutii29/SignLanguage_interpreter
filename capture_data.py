import os
import cv2

data_dir = './dataset'
limit = 200

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Initialize camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Get input from user
    alphabet = input("Enter alphabet to collect images (press -1 to stop): ")
    
    # Stop the session if the user enters '-1'
    if alphabet == '-1':
        break
    
    # Create folder for respective alphabet if it doesn't exist
    alphabet_dir = os.path.join(data_dir, str(alphabet))
    if not os.path.exists(alphabet_dir):
        os.makedirs(alphabet_dir)
    
    # Count number of existing images to avoid overwriting to it
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

        # Display instruction and ROI rectangle
        cv2.putText(frame, 'Press # to start', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)

        cv2.imshow('frame', frame)
        
        # Enter '#' to start capturing imagess
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

        # Display ROI rectangle
        cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)

        # Crop the ROI
        cropframe = frame[40:300, 0:300]

        # Convert to grayscale and resize for model input
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (50, 50))

        # Show the mirrored frame with the ROI rectangle
        cv2.imshow('frame', frame)
        
        # Display the cropped ROI in a separate window
        cv2.imshow('ROI', cropframe)

        cv2.waitKey(40)

        # Save cropped ROI image
        cv2.imwrite(os.path.join(alphabet_dir, '{}.jpg'.format(counter)), cropframe)
        counter += 1
    
    # Acknowledge
    print(f"Done collecting data for {alphabet}")

# Release camera
camera.release()

# End current session
cv2.destroyAllWindows()
