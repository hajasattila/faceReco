import cv2

# Open the video stream
video_capture = cv2.VideoCapture("video.mp4")

# Initialize variables
total_frames = 0
frames_with_face = 0

# Process each frame in the video
while True:
    # Read the next frame from the video
    ret, frame = video_capture.read()

    # If there are no more frames to process, break out of the loop
    if not ret:
        break

    # Increment the total number of frames
    total_frames += 1

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's Haar cascade classifier to detect faces
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # If at least one face was detected, increment the count of frames with faces
    if len(faces) > 0:
        frames_with_face += 1

# Calculate and print the percentage of frames with faces
face_percentage = frames_with_face / total_frames * 100
print(f"{face_percentage:.2f}% of the frames contain a human face.")
