import cv2
import mediapipe as mp
import time


mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw=mp.solutions.drawing_utils
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=2)
pTime=0

cap=cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    #print(results)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,facelms)


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,120,23),3)
    


    cv2.imshow('Camera',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break