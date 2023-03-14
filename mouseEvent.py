import cv2
'''img=cv2.imread('hero.jpg')

def click_mouse_events(event,a,b,flags,params):
    font=cv2.FONT_HERSHEY_COMPLEX
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(img,str(a)+','+str(b),(a,b),font,2,(0,0,255),3)
        cv2.imshow('image',img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.setMouseCallback('image',click_mouse_events)
cv2.destroyAllWindows()'''

'''import cv2
img=cv2.imread('hero.jpg')
_,thresh_binary=cv2.threshold(img,200,255,cv2.THRESH_BINARY)
#thresh_binary matra haina,inverse_binary,trunc,t0Zero,inversetozero
cv2.imshow('original',img)
cv2.imshow('thresh_binary',thresh_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''import matplotlib.pyplot as plt
img=cv2.imread('hero.jpg')
cv2.imshow('hero',img)
plt.imshow(img)
plt.show()
cv2.waitKey(0)'''

'''-----------------------------ADAPTIVE THRESHOLDING-------------------------------------------'''
img=cv2.imread('hero.jpg',0)
adaptive_threshold=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
cv2.imshow('original',img)
cv2.imshow('adaptive',adaptive_threshold)
cv2.waitKey(0)
