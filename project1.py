import cv2
import mediapipe as mp 
import time


cap = cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

pTime=0
cTime=0
while True:
    success,img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id,lms in enumerate(handlms.landmark):
               # print(id,lms)
                h,w,c=img.shape
                cx,cy=int(lms.x*w),int(lms.y*h)
                print(id,cx,cy)
                #if id==4:
                cv2.circle(img,(cx,cy),15,(250,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,250),2)
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
     break
    
   

cv2.destroyAllWindows()