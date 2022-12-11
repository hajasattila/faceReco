import cv2

#Betoltjük a cascade frontalfacet
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# A videó amit elemezni akarunk
cap = cv2.VideoCapture('video.mp4')

# scaling faktor
scaling_factor = 0.5

# arcok amit érzékelt
num_faces = 0

# itratívan átnézi az egész videót
while True:
    # beolvassa a következő képkockákat
    ret, frame = cap.read()

    

    # resizeolja a framet
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # a képkockát grayscaleli
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # futtatja az arc érzékelő cascadet, a grayscale frameken
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # inkrementálja az arcokat, amit érzékelt
    num_faces += len(faces)

    # négyzetet rajzol az arcok körül
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Vége van, ha nincs több frame.
    if not ret:
        break

# kiírja mennyi arcot érzékelt
print("Number of faces detected:", num_faces)

# elengedi a capturelt objektumot
cap.release()
