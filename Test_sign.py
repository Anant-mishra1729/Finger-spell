from nbformat import read
from tqdm import trange
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
		filedict.append((pickle.load(handle),file[-8]))

def findDistances(positions):
	distMatrices = []
	for k in range(len(positions)):
		hand = positions[k]
		distMatrix = np.zeros([len(hand), len(hand)], dtype='float')
		palm_dist = ((hand[0][0]-hand[3][0])**2 +
					 (hand[0][1]-hand[3][1])**2)**0.5
		for i in range(0, len(hand)):
			for j in range(0, len(hand)):
				distMatrix[i][j] = (
					((hand[i][0] - hand[j][0])**2 + (hand[i][1] - hand[j][1])**2)**0.5)/palm_dist
		distMatrices.append(distMatrix)
	return distMatrices


def findError(known,test):
	error = 0
	for i in range(0,len(known)):
		for j in range(0,len(test)):
			error += abs(known[i][j] - test[i][j]) 
	return int(error)

# test_image = cv2.imread("Resources\\ISL\\Indian\\A\\7.jpg")
# results = tracker.findHands(cv2.cvtColor(test_image,cv2.COLOR_RGB2BGR))
# if results:
# 	positions = []
# 	for landmarks in results:
# 		positions.append(tracker.getPos(landmarks.landmark,test_image.shape,points))
# 	test = findDistances(positions)

count = 0
while True:
	success, image = cap.read()
	results = tracker.findHands(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	positions = []
	if results:
		for landmarks in results:
			tracker.drawHands(image, landmarks)
			positions.append(tracker.getPos(landmarks.landmark, image.shape, points))
		test = findDistances(positions)
		for gesture in filedict:
			if len(gesture[0]) == len(test):
				for i in range(len(gesture[0])):
					if findError(gesture[0][i],test[i]) < 20:
						print(gesture[1],count)
						count += 1

	cv2.imshow("Output", cv2.flip(image,1))
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
