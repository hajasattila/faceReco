import cv2

#Betöltjük a Haar cascade filet, hogy az arcokat tudja a python érzékelni
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Inicializáljuk a videót captureló objectet
cap = cv2.VideoCapture('video.mp4')

# Változók, amiben számolni fogjuk a max framet, az arcokat felismert frameket, és a %-ot majd a harmadikban számoljuk ki
num_frames = 0
num_faces = 0
num_percent = 0

# Keep looping until the user hits the 'q' key
#Loopolunk addig, amig "Q"-t nem nyomunk, vagy amíg nem találjuk meg az utolsó képkockát

while True:
    #beolvassuk a következő képkockát a videóból
    ret, frame = cap.read()
    #amíg megy a videó, addig a framek is növekednek
    num_frames += 1
    #konvertáljuk a framet grayscaleba
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Érzékeljuk az arcota jelenegi frameban
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #Kék négyzet az arc körül, és növeljük a cikluson belül az arc számát
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        num_faces += 1
        
    # Megmutatjuk az usernak a videót
    cv2.imshow('video', frame)
    # Kiszámoljuk %-ban a jelenlegi értéket
    num_percent = (1-(num_faces/num_frames))*100
    #Ellenőrizzük, hogy megnyomtuk-e a "q"-t
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #Ha nincs következő frame, akkor brake
    if not ret:
        break
    
    #Mindig frissen kiírjuk, hogy hány %-a a videóna a felismert arc
    print(f'{num_percent:.2f}%-ban volt arc a videóban.')
    
#Felszabadítjuk a videós objectumot
cap.release()
