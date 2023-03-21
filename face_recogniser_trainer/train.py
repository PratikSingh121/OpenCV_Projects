import os
import cv2 as cv
import numpy as np

BASE_DIR = os.getcwd()
FACE_DIR = os.path.join(BASE_DIR, 'Faces')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
	for person in os.listdir(FACE_DIR):
		if person == 'indexCount.txt':
			continue
		person_path = os.path.join(FACE_DIR, person)
		with open(os.path.join(person_path, 'index.txt'),'r') as f:
			label = int(f.read())

		for img in os.listdir(person_path):
			if img == 'index.txt':
				continue
			img_path = os.path.join(person_path, img)
			print(img_path)
			img_array = cv.imread(img_path)
			gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
			face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

			for (x,y,w,h) in face_rect:
				face_roi = gray[y:y+h, x:x+w]
				features.append(face_roi)
				labels.append(label)

create_train()
print('Training done...')

features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('training_data/face_trained.yml')
np.save('training_data/features.npy', features)
np.save('training_data/labels.npy', labels)
print('Training saved...')