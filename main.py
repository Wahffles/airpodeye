# FRC 868 Vision Code Testing Script

# Playing around with opencv to try to learn it

import cv2 
import numpy as np
import time
import notacopy
from tkinter import *
from threading import *
pTime = 0

cam = cv2.VideoCapture(0)
if cam.isOpened() == False:
    print("Error: Camera is not open.")
    exit()

max_value = 255
max_value_H = 360//2

notacopy.load_constants()

minH = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[0])
minS = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[1])
minV = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[2])
maxH = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[0])
maxS = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[1])
maxV = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[2])

# colors

#
def tkinterwin():
    global minH
    global minS
    global minV
    global maxH
    global maxS
    global maxV
    
    techblue = "#0F64FA"
    techgold = "#FAC805"
    dblue = "#0C50C7"
    dgold = "#C79F04"
    lblue = "#5A94FB"
    lgold = "#FBD850"
    window = Tk()
    window.title("AirPodEYE Calibration Interface")
    window.geometry('600x500')

    notacopy.load_constants()

    frame = Frame(window, bg=lblue)
    frame.pack(side="top", expand=True, fill="both")

    minnH = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Min Hue", font="Poppins 10 bold")
    minnH.pack(pady=2)
    minnH.set(minH)
    maxxH = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Max Hue", font="Poppins 10 bold")
    maxxH.pack(pady=2)
    maxxH.set(maxH)
    minnS = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Min Saturation", font="Poppins 10 bold")
    minnS.pack(pady=2)
    minnS.set(minS)
    maxxS = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Max Saturation", font="Poppins 10 bold")
    maxxS.pack(pady=2)
    maxxS.set(maxS)
    minnV = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Min Value", font="Poppins 10 bold")
    minnV.pack(pady=2)
    minnV.set(minV)
    maxxV = Scale(frame, bg=techblue, fg=techgold, troughcolor=techgold, length=500, from_=0, to=255, orient=HORIZONTAL, label="Max Value", font="Poppins 10 bold")
    maxxV.pack(pady=2)
    maxxV.set(maxV)

    minH = minnH.get()
    minS = minnS.get()
    minV = minnV.get()
    maxH = maxxH.get()
    maxS = maxxS.get()
    maxV = maxxV.get()

    notacopy.HSV_BOUNDS.MAIN_BOUND_L[0] = minH
    notacopy.HSV_BOUNDS.MAIN_BOUND_L[1] = minS
    notacopy.HSV_BOUNDS.MAIN_BOUND_L[2] = minV
    notacopy.HSV_BOUNDS.MAIN_BOUND_U[0] = maxH
    notacopy.HSV_BOUNDS.MAIN_BOUND_U[1] = maxS
    notacopy.HSV_BOUNDS.MAIN_BOUND_U[2] = maxV

    save = Button(frame, text="Save Values", bg=dblue, fg=lgold, width=35, font="Poppins 9 bold", command=notacopy.dump_constants)
    save.pack(pady=5)
    window.config()
    window.mainloop()

def cvwin():
    global pTime
    global minH
    global minS
    global minV
    global maxH
    global maxS
    global maxV

    notacopy.load_constants()

    minH = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[0])
    minS = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[1])
    minV = int(notacopy.HSV_BOUNDS.MAIN_BOUND_L[2])
    maxH = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[0])
    maxS = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[1])
    maxV = int(notacopy.HSV_BOUNDS.MAIN_BOUND_U[2])
    
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

t1 = Thread(target=tkinterwin)
t2 = Thread(target=cvwin)

t1.start()
t2.start()