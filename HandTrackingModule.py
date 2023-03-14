import cv2
import mediapipe as mp 
import time


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self. mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result=self.hands.process(imgRGB)
        print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                if draw:
                 self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)

        return img      


            




''' for id,lms in enumerate(handlms.landmark):
                # print(id,lms)
                h,w,c=img.shape
                cx,cy=int(lms.x*w),int(lms.y*h)
                print(id,cx,cy)
                #if id==4:
                cv2.circle(img,(cx,cy),15,(250,0,255),cv2.FILLED)'''
   
    
   




def main():

    
  pTime=0
  cTime=0
  cap=cv2.VideoCapture(0)
  detector = handDetector()
  while True:
      success,img = cap.read()
      img=detector.findHands(img)


      cTime=time.time()
      fps=1/(cTime-pTime)
      pTime=cTime

      cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,250),2)
      cv2.imshow('webcam',img)

      if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    

if __name__=="__main__":
    main()