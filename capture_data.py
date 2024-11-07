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
    
    # Count number of existing images to avoid overwriting on it
    existing_images = len(os.listdir(alphabet_dir))

    print(f'Collecting data for alphabet {alphabet}')
    
    # ROI (Region of Interest)
    roi_top_left_x = 0
    roi_top_left_y = 50
    roi_width = 350
    roi_height = 350

    # Wait till '#' is entered
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("Error: unable to capture image")
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Display instruction and ROI rectangle
        cv2.putText(frame, 'Press # to start', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv2.rectangle(frame, (roi_top_left_x, roi_top_left_y), (roi_top_left_x + roi_width, roi_top_left_y + roi_height), (0, 255, 0), 2)

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
        cv2.rectangle(frame, (roi_top_left_x, roi_top_left_y), (roi_top_left_x + roi_width, roi_top_left_y + roi_height), (0, 255, 0), 2)

        # Crop the ROI
        roi = frame[roi_top_left_y:roi_top_left_y + roi_height, roi_top_left_x:roi_top_left_x + roi_width]

	    # Convert to grayscale
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	
	    # Apply Gaussian Blur (optional)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # Resize to 50x50 pixels
        roi = cv2.resize(roi, (50, 50))
        
        # Show the mirrored frame with the ROI rectangle
        cv2.imshow('frame', frame)
        
        # Display the cropped ROI in a separate window
        cv2.imshow('ROI', roi)

        cv2.waitKey(40)

        # Save cropped ROI image
        cv2.imwrite(os.path.join(alphabet_dir, '{}.jpg'.format(counter)), roi)
        counter += 1
    
    # Display 'Done' message
    print(f"Done collecting data for {alphabet}")

# Release camera
camera.release()

# End current session
cv2.destroyAllWindows()
