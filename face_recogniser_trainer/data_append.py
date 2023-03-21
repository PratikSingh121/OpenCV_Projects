import os
import cv2 as cv
import time

BASE_DIR = os.getcwd()
FACE_DIR = os.path.join(BASE_DIR, 'Faces')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
indexExist = 0

person = input('Enter Your Name.>')
while True:
	b=0
	person_path = os.path.join(FACE_DIR, person)
	if os.path.exists(person_path):
		if input('Person Already exist,Add new data to the same person? (Y/N) >').lower() == 'y':
			indexExist = 1
			b=1
		else:
			person = input('Person data already exists,Enter an Alternative name: >')
			person_path = os.path.join(FACE_DIR, person)
	if b == 1:
		break
	else:
		break
os.makedirs(person_path, exist_ok = True)
if indexExist == 0:
	with open(os.path.join(FACE_DIR,'indexCount.txt'),'r') as f:
		indexCount = int(f.read())
	with open(os.path.join(person_path,'index.txt'),'w') as f:
		f.write(str(indexCount))
	with open(os.path.join(FACE_DIR,'indexCount.txt'),'w') as f:
		f.write(f'{indexCount + 1}')	
existing_images = len(os.listdir(person_path))-1

counter = existing_images

def img_capture(counter = counter):
	capture = cv.VideoCapture(0)
	while True:
		isTrue, img = capture.read()
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		face_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

		for (x,y,w,h) in face_rect:
			cv.rectangle(img, (x, y), (x+w,y+h), (0,255,0), thickness=3)
			cv.putText(img, 'Studing Face', (x,y+h+40), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
		cv.imshow('Detected Faces',img)

		if len(face_rect)>0:
			cv.imwrite(os.path.join(person_path,f'{counter}.jpg'), gray)
			counter += 1
		key = cv.waitKey(1)
		if key == ord('e'):
			capture.release()
			break

print('Starting Camera in 5 sec.')
time.sleep(5)
print('Camera Started...')
img_capture()
cv.destroyAllWindows()
print('Data added:Thank you')

if input('\n Do you wish to train data now?(Y/N) >').lower()=='y':
	os.system('python train.py')
else:
	print('Exiting...')