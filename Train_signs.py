from unittest import result
from libs.HandTracker import HandTracker
from libs import utils
import cv2
import numpy as np
import pickle

def findDistances(positions):
	distMatrices = []
	for k in range(len(positions)):
		hand = positions[k]
		distMatrix = np.zeros((len(hand),len(hand)),dtype = 'float')
		palm_dist = ((hand[0][0]-hand[3][0])**2 + (hand[0][1]-hand[3][1])**2)**0.5
		for i in range(0,len(hand)):
			for j in range(0,len(hand)):
				distMatrix[i][j] = (((hand[i][0] - hand[j][0])**2 + (hand[i][1] - hand[j][1])**2)**0.5)/palm_dist
		distMatrices.append(distMatrix)
	return distMatrices


image_dir = "Resources\\ISL\\Indian\\2"
output_path = "Output"

images = utils.getFiles(image_dir,(".jpg",".png"))

points = [0,4,5,9,13,17,8,12,16,20]
tracker = HandTracker(static = True)
error = 0
avg = np.zeros([len(points),len(points)],dtype = 'float')

for img in images:
	image = cv2.imread(img)
	results = tracker.findHands(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
	positions = []
	if results:
		for landmarks in results:
			tracker.drawHands(image,landmarks)
			positions.append(tracker.getPos(landmarks.landmark,image.shape,points))
		result = findDistances(positions)
	cv2.imshow("Output",image)
	if cv2.waitKey(0) & 0xff == ord('q'):
		break	

with open(f'{output_path}\\{image_dir[-1]}.pickle','wb') as handle:
	pickle.dump(result,handle)
