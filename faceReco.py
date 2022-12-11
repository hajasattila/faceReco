import cv2

# Create a new named window
cv2.namedWindow("Face Detection")

# Start capturing video from the webcamera
video_capture = cv2.VideoCapture(0)

# Create a Haar Cascade object for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loop indefinitely
while True:
    # Read the current frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the current frame in the named window
    cv2.imshow("Face Detection", frame)

    # Wait for a key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture object
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()
