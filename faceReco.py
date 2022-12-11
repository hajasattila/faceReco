import cv2

# initialize the video stream and the face detector
vs = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize variables to keep track of the total number of frames and the number of frames with a face
total_frames = 0
face_frames = 0

# loop until the end of the video stream is reached
while True:
    # read the next frame from the stream
    ret, frame = vs.read()

    # if the frame could not be read, break out of the loop
    if not ret:
        break

    # increment the total number of frames
    total_frames += 1

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the frame
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # if at least one face was detected, increment the face_frames counter
    if len(faces) > 0:
        face_frames += 1

    # display the current frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break out of the loop
    if key == ord("q"):
        break

# calculate the percentage of frames with a face
face_percentage = (face_frames / total_frames) * 100

# print the percentage of frames with a face
print("{:.2f}% of frames contained a face".format(face_percentage))

# release the video stream
vs.release()

# close all windows
cv2.destroyAllWindows()
