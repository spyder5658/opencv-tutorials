import cv2
import numpy as np
'''print('package imported')

img = cv2.imread('car.jpg')
cv2.imshow('otuput image',img)
cv2.waitKey(0)'''

'''cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
while True:
    success,img = cap.read()
    cv2.imshow('Output',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break'''

'''img = cv2.imread('car.jpg')
kernel = np.ones((5,5),np.uint8)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurr_img=cv2.GaussianBlur(gray_img,(7,7),0)
can_img=cv2.Canny(img,100,100)
dilated = cv2.dilate(can_img,kernel,iterations=1)
eroded = cv2.erode(dilated,kernel,iterations=1)
while True:
 cv2.imshow('output',gray_img)
 cv2.imshow('blurr image',blurr_img)
 cv2.imshow('Canny image',can_img)
 cv2.imshow('dilated image',dilated)
 cv2.imshow('eroded image',eroded)
 if cv2.waitKey(0) & 0xFF == ord('q'):
   break'''

'''img=cv2.imread('car.jpg')
print(img.shape)

re_img=cv2.resize(img,(200,100))

cv2.imshow('original',img)
cv2.imshow('resized',re_img)
cv2.waitKey(0)'''

'''img=np.zeros((500,500,3),np.uint8)
#Blue screen
#img[:]=255,0,0
#green
#img[:]=0,255,0
#red
#img[:]=0,0,255
print(img)
#cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(245,45,0),3)
#cv2.rectangle(img,(0,0),(200,120),(230,120,122),2)
#cv2.circle(img,(250,120),50,(200,213,123),cv2.FILLED)
cv2.putText(img,'I love You Naakbali',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(120,130,35))

cv2.imshow('screen',img)
cv2.waitKey(0)'''
def empty(a):
  pass

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",35,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",56,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",139,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
  img=cv2.imread('car.jpg')
  HSV_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  stak=np.hstack((img,HSV_img))
  h_min=cv2.getTrackbarPos('Hue Min','TrackBars')
  h_max=cv2.getTrackbarPos('Hue Max','TrackBars')
  s_min=cv2.getTrackbarPos('Sat Min','TrackBars')
  s_max=cv2.getTrackbarPos('Sat Max','TrackBars')
  v_min=cv2.getTrackbarPos('Val Min','TrackBars')
  v_max=cv2.getTrackbarPos('Val Max','TrackBars')
  print(h_min,h_max,s_min,s_max,v_min,v_max)
  lower=np.array([h_min,s_min,v_min])
  upper=np.array([h_max,s_max,v_max])
  mask = cv2.inRange(HSV_img,lower,upper)
  result=cv2.bitwise_and(img,img,mask=mask)

  cv2.imshow('output',stak)
  cv2.imshow("mask",mask)
  cv2.imshow('result image',result)
  cv2.waitKey(1)

'''def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #print(area)
        if area:
         cv2.drawContours(imgcntr,cnt,-1,(255,0,0),3)
         peri=cv2.arcLength(cnt,True)
         #print(peri)
         approx=cv2.approxPolyDP(cnt,0.02*peri,True) 
         objCOr=len(approx)
         x,y,w,h=cv2.boundingRect(approx)

         if objCOr==3: objectType="Tri" 
         else:objectType="None"

         cv2.rectangle(imgcntr,(x,y),(x+w,y+h),(0,255,0),3)
         cv2.putText(imgcntr,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)

img = cv2.imread('shapes.png')
imgcntr=img.copy()
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(imggray,(7,7),1)
canny=cv2.Canny(blur,50,50)
getContours(canny)
black = np.zeros_like(img)
imgStack=stackImages(0.8,([img,imggray,blur],[canny,imgcntr,black]))
cv2.imshow('stack',imgStack)

cv2.waitKey(0)'''

'''faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread('hero.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces:
    cv2. rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('output',img)
cv2.waitKey(0)'''

