import cv2
import mediapipe as mp
import time


mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraws=mp.solutions.drawing_utils
pTime=0
cap = cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraws.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,120,23),3)

    cv2.imshow('camera',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break