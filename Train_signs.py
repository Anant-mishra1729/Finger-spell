from unittest import result
from libs.HandTracker import HandTracker
from libs import utils
import cv2
import numpy as np
import pickle

def eucledianDistances(handData,palmdist):
	""" Eucledian distances between points 0,4,5,9,13,17,8,12,16 and 20 """
	distMatrix = np.zeros([len(handData),len(handData)],dtype = 'float')
	pd = palmdist[0]
	for i in range(len(handData)):
		if (i > 9):
			pd = palmdist[1]
		for j in range(len(handData)):
			distMatrix[i][j] = (((handData[i][0]-handData[j][0])**2 + (handData[i][1]-handData[j][1])**2)**0.5)/pd
	return distMatrix


image_dir = "Resources\\ISL\\Indian\\Thank you"
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
	palmdist = []
	if results:
		for landmarks in results:
			tracker.drawHands(image,landmarks)
			position = tracker.getPos(landmarks.landmark,image.shape,points)
			pd = ((position[0][0]-position[3][0])**2 + (position[0][1]-position[3][1])**2)**0.5
			palmdist.append(pd)
			print(palmdist)
			positions.extend(position)
		result = (eucledianDistances(positions,palmdist),(image_dir.split("\\"))[-1])
		print(result)
	cv2.imshow("Output",image)
	if cv2.waitKey(0) & 0xff == ord('q'):
		break	

with open('{}\\{}.pickle'.format(output_path,(image_dir.split("\\"))[-1]),'wb') as handle:
	pickle.dump(result,handle)
	print("Done...")
