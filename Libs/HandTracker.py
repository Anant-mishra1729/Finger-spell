import mediapipe as mp
class HandTracker:
	def __init__(self,static = False,max_hands = 2,dconf = 0.5,tconf = 0.5):
		self.hands_module = mp.solutions.mediapipe.python.solutions.hands
		self.draw_module = mp.solutions.mediapipe.python.solutions.drawing_utils
		self.hands = self.hands_module.Hands(static_image_mode= static,max_num_hands=max_hands,min_detection_confidence=dconf,min_tracking_confidence=tconf)
	
	def findHands(self,image):
		return self.hands.process(image).multi_hand_landmarks
	
	def drawHands(self,image,landmarks):
		self.draw_module.draw_landmarks(image,landmarks,self.hands_module.HAND_CONNECTIONS)

	def getPos(self,landmark,shape,points = []):
		lmList = list(landmark)
		height,width = shape[0],shape[1]
		if not len(points):
			return [(int(lmList[i].x*width),int(lmList[i].y*height)) for i in range(len(lmList))]
		else:
			return [(int(lmList[i].x*width),int(lmList[i].y*height)) for i in points]

def main():
	import cv2
	import time
	# Camera configuration
	width = 640
	height = 320
	cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
	cap.set(cv2.CAP_PROP_FPS,30)
	cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
	pTime = 0
	hands = HandTracker()

	while True:
		success,image = cap.read()
		imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		results = hands.findHands(imageRGB)
		if results:
			for landmarks in results:
				hands.drawHands(image,landmarks)
				pos = hands.getPos(landmarks.landmark,image.shape,[4])
				cv2.circle(image,pos[0],15,(0,0,255),cv2.FILLED)
		
		#FPS
		cTime = time.time()
		fps = 1/(cTime-pTime)
		pTime = cTime
		cv2.putText(image,str(int(fps)),(600,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
		
		cv2.imshow("Output",image)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

if __name__ == '__main__':
	main()