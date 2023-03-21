import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

BASE_DIR = os.getcwd()
FACE_DIR = os.path.join(BASE_DIR, 'Faces')

LABELS = {}
for people in os.listdir(FACE_DIR):
	if people == 'indexCount.txt':
		continue
	with open(os.path.join(FACE_DIR, people,'index.txt'),'r') as f:
		LABELS[int(f.read())] = people

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('training_data/face_trained.yml')

capture = cv.VideoCapture(0)
capture.set(3, 800)
capture.set(4, 600)
while True:
	isTrue, img = capture.read()

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	face_rect = haar_cascade.detectMultiScale(gray,1.1,4)
	if len(face_rect)!=0:
		for (x,y,w,h) in face_rect:
			face_roi = gray[y:y+h , x:x+w]

			label, confidence = face_recognizer.predict(face_roi)
			#print(f'Label = {LABELS[label]} with a confidence of {confidence}') 
			cv.putText(img, str(LABELS[label]), (x,y+h+40), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
			cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness = 2)

			cv.imshow('Faces', img)
	else:
		cv.imshow('Faces',img)

	key = cv.waitKey(1)

	if key == ord('e'):
		break
	if key == ord('c'):
		cv.imwrite('../photos/me.jpg', img)