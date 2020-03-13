import numpy as np
import cv2
import imutils
import time

#lower and upper boundaries of color red
redLower = np.array([161, 155, 84]) 
redUpper = np.array([179, 255, 255])

video_cap = cv2.VideoCapture(0)

#give cam a second
time.sleep(1.0)

while True:
	_, frame = video_cap.read()
	if frame is None:
		break
	# resize the frame
	frame = imutils.resize(frame, width = 600)
	#bluring frame
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	#convert frame to the HSV
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	#mask for the color red
	mask = cv2.inRange(hsv, redLower, redUpper)
	
	#remove any small blobs left in the mask
	mask = cv2.erode(mask, None, iterations = 2)
	mask = cv2.dilate(mask, None, iterations = 2)

	#find contours in the mask and initialize the current
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	
	#if at least one contour was found
	if len(cnts) != 0:
		#find the largest contour in the mask, 
		c = max(cnts, key=cv2.contourArea)
		#compute the minimum enclosing circle and centroid, (x, y) center of the ball
		((x, y), r) = cv2.minEnclosingCircle(c)
		m = cv2.moments(c)
		center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
		if r > 10:
			#draw the circle and centroid
			cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
video_cap.release()
cv2.destroyAllWindows()
