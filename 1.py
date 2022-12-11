import cv2

# load a cascade filet (face érzékeléshez)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# videó fike egnyitása
video = cv2.VideoCapture('video.mp4')

# total frames a videóban
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# arc amit látott a progi
face_frames = 0

# minden framet beolvasunk
while video.isOpened():
    ret, frame = video.read()

    # ha valamit nem jól olvastunk be, kilép a ciklusból
    """ if not ret:
        break """

    # a frameket grayscaleba átkonvertáljuk
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # felsimerjuk az arcokat
    faces = face_cascade.detectMultiScale(gray)

    # ha legalább egy arcot látott  videó, akkor számoljuk azt
    if len(faces) > 0:
        face_frames += 1

# kiszámoljuk mennyi az annyi
face_percentage = (face_frames / total_frames) * 100

# kiiratás
print(f'{face_percentage:.2f}% of the video consists of human faces.')
