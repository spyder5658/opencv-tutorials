import cv2
import mediapipe as mp
import time
mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()
mpDraw=mp.solutions.drawing_utils
pTime=0

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    #print(results.detections)
    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),int(bboxC.width*iw),int(bboxC.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
        
            
    


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,120,23),3)
    

    cv2.imshow('Camera',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break