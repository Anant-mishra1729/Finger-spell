from os import walk,path
import cv2
import time

pTime = 0
hTime = 4
def getFiles(filePath,exts):
	imgPaths = []
	for (root, dirs, files) in walk(filePath):
		for f in files:
			if path.splitext(f)[1] in exts:
				imgPaths.append(path.join(root, f))
	return imgPaths

def camera(width,height):
	cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
	cap.set(cv2.CAP_PROP_FPS,30)
	cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
	return cap

def Fps(image):
	global pTime
	cTime = time.time()
	fps = 1/(cTime-pTime)
	pTime = cTime
	cv2.putText(image,str(int(fps)),(600,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
