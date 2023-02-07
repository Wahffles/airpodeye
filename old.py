# FRC 868 Vision Code Testing Script

# Playing around with opencv to try to learn it

import cv2 
import numpy as np
import time
import argparse
import notacopy
pTime = 0

cam = cv2.VideoCapture(0)
if cam.isOpened() == False:
    print("Error: Camera is not open.")
    exit()

max_value = 255
max_value_H = 360//2

notacopy.load_constants()

minH = notacopy.HSV_BOUNDS.MAIN_BOUND_L[0]
minS = notacopy.HSV_BOUNDS.MAIN_BOUND_L[1]
minV = notacopy.HSV_BOUNDS.MAIN_BOUND_L[2]
maxH = notacopy.HSV_BOUNDS.MAIN_BOUND_U[0]
maxS = notacopy.HSV_BOUNDS.MAIN_BOUND_U[1]
maxV = notacopy.HSV_BOUNDS.MAIN_BOUND_U[2]

def lowHtrack(val):
    global minH
    global maxH
    minH = val
    minH = min(maxH-1, minH)
    cv2.setTrackbarPos("min H", "Slidebars", minH)

def highHtrack(val):
    global minH
    global maxH
    maxH = val
    maxH = max(maxH, minH+1)
    cv2.setTrackbarPos("max H", "Slidebars", maxH)

def lowStrack(val):
    global minS
    global maxS
    minS = val
    minS = min(maxS-1, minS)
    cv2.setTrackbarPos("min S", "Slidebars", minS)

def highStrack(val):
    global minS
    global maxS
    maxS = val
    maxS = max(maxS, minS+1)
    cv2.setTrackbarPos("max S", "Slidebars", maxS)

def lowVtrack(val):
    global minV
    global maxV
    minV = val
    minV = min(maxV-1, minV)
    cv2.setTrackbarPos("min V", "Slidebars", minV)

def highVtrack(val):
    global minV
    global maxV
    maxV = val
    maxV = max(maxV, minV+1)
    cv2.setTrackbarPos("max V", "Slidebars", maxV)


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv2.VideoCapture(args.camera)
cv2.namedWindow("Slidebars")

# trackbars
cv2.createTrackbar("MIN HUE", "Slidebars" , minH, max_value_H, lowHtrack)
cv2.createTrackbar("MAX HUE", "Slidebars" , maxH, max_value_H, highHtrack)
cv2.createTrackbar("MIN SATURATION", "Slidebars" , minS, max_value, lowStrack)
cv2.createTrackbar("MAX SATURATION", "Slidebars" , maxS, max_value, highStrack)
cv2.createTrackbar("MIN VALUE", "Slidebars" , minV, max_value, lowVtrack)
cv2.createTrackbar("MAX VALUE", "Slidebars" , maxV, max_value, highVtrack)


while True:
    ret, frame = cam.read()
    if ret == False:
        print("Error: Cannot read frame.")
        break

    # convert og to hsv, then filter hsv to create binary image
    frame = cv2.GaussianBlur(frame, (7, 7), sigmaX=1.5, sigmaY=1.5)
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (minH, minS, minV), (maxH, maxS, maxV))

    # post processing stuff
    # dilate/erode, contours

    kernel = np.ones((5, 5), np.uint8)
    frame_threshold = cv2.erode(frame_threshold, kernel, iterations=3)
    frame_threshold = cv2.dilate(frame_threshold, kernel, iterations=2)
    contours = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(frame_threshold)
        conts += [(x,y,w,h)]
    for x,y,w,h in conts:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("Object Detection", frame_threshold)
    
    
    if cv2.waitKey(1) == ord('q'):
        notacopy.HSV_BOUNDS.MAIN_BOUND_L[0] = minH
        notacopy.HSV_BOUNDS.MAIN_BOUND_L[1] = minS
        notacopy.HSV_BOUNDS.MAIN_BOUND_L[2] = minV
        notacopy.HSV_BOUNDS.MAIN_BOUND_U[0] = maxH
        notacopy.HSV_BOUNDS.MAIN_BOUND_U[1] = maxS
        notacopy.HSV_BOUNDS.MAIN_BOUND_U[2] = maxV
        notacopy.dump_constants()
        break                                                                               

cam.release()
cv2.destroyAllWindows()