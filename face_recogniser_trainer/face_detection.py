#Just detecting Face
#Classifier is used.Algorithm that detects whether a face is present or not.
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

capture = cv.VideoCapture(0)

while True:
	isTrue, img = capture.read()
	img = cv.flip(img ,1)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 4)
	print(f'Number of faces found = {len(face_rect)}')

	for (x,y,w,h) in face_rect:
		cv.rectangle(img, (x, y), (x+w,y+h), (0,255,0), thickness=3)

	cv.imshow('Detected Faces',img)
	key = cv.waitKey(1)
	if key == ord('e'):
		break
