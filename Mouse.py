import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.8,maxHands=2)

video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    hands,img=detector.findHands(frame)
    if hands:
        lmlist=hands[0]
        fingerUP=detector.fingersUp(lmlist)
        if fingerUP==[0,0,0,0,0]:
            cv2.putText(frame,'Finger Count is 0',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1)      
        if fingerUP==[0,1,0,0,0]:
            cv2.putText(frame,'Finger Count is 1',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1)
        if fingerUP==[0,1,1,0,0]:
            cv2.putText(frame,'Finger Count is 2',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1) 
        if fingerUP==[0,1,1,1,0]:
            cv2.putText(frame,'Finger Count is 3',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1)      
        if fingerUP==[0,1,1,1,1]:
            cv2.putText(frame,'Finger Count is 4',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1)  
        if fingerUP==[1,1,1,1,1]:
            cv2.putText(frame,'Finger Count is 5',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,240,123),1)                      
    cv2.imshow('camera',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break