from libs.HandTracker import HandTracker
from libs.utils import camera, getFiles
import cv2
import numpy as np
import pickle

tracker = HandTracker()
cap = camera(640, 320)
files = getFiles("Output", (".pickle"))
points = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]

filedict = []
for file in files:
	with open(file,'rb') as handle:
		filedict.append(pickle.load(handle))

def findDistances(handData,palmdist):
	distMatrix = np.zeros([len(handData),len(handData)],dtype = 'float')
	pd = palmdist[0]
	for i in range(len(handData)):
		if (i > 9):
			pd = palmdist[1]
		for j in range(len(handData)):
			distMatrix[i][j] = (((handData[i][0]-handData[j][0])**2 + (handData[i][1]-handData[j][1])**2)**0.5)/pd
	return distMatrix


def findError(known,test):
	error = 0
	for i in range(0,len(known)):
		for j in range(0,len(test)):
			error += abs(known[i][j] - test[i][j]) 
	return int(error)

# test_image = cv2.imread("Resources\\ISL\\Indian\\1\\1.jpg")
# results = tracker.findHands(cv2.cvtColor(test_image,cv2.COLOR_RGB2BGR))
# if results:
# 	positions = []
# 	palmdist = []
# 	for landmarks in results:
# 		position = tracker.getPos(landmarks.landmark,test_image.shape,points)
# 		pd = ((position[0][0]-position[3][0])**2 + (position[0][1]-position[3][1])**2)**0.5
# 		palmdist.append(pd)
# 		positions.extend(position)
# 	test = findDistances(positions,palmdist)
# known = (filedict[0])[0]
# print(findError(known,test))


count = 0
while True:
	success, image = cap.read()
	image = cv2.flip(image,1)
	results = tracker.findHands(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	positions = []
	palmdist = []
	if results:
		for landmarks in results:
			# tracker.drawHands(image,landmarks)
			position = tracker.getPos(landmarks.landmark,image.shape,points)
			palmdist.append(((position[0][0]-position[3][0])**2 + (position[0][1]-position[3][1])**2)**0.5)
			positions.extend(position)
		test = findDistances(positions,palmdist)
		err_matrix = []
		for gesture in filedict:
			if len(gesture[0]) == len(test):
				err_matrix.append((findError(gesture[0],test),gesture[1]))
		if len(err_matrix):
			cv2.putText(image,min(err_matrix)[1],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
	cv2.imshow("Output", image)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
