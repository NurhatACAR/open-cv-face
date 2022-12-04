import cv2

yuz = cv2.CascadeClassifier(r"classifier/haarcascade_frontalface_default.xml")

foto = cv2.imread(r"photos/eaffb374borat.jpg")

gray = cv2.cvtColor(foto,cv2.COLOR_BGR2GRAY)

faces = yuz.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(foto,(x,y),(x+w, y+h), (0,255,0),3)
    
cv2.imshow('img',foto)
cv2.waitKey(0)