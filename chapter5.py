'''warp perspective'''
import cv2
import numpy as np
img=cv2.imread('cards.jpg')
width,height=250,350
pnts1=np.float32([[313,69],[493,131],[206,386],[383,457]])
pnts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pnts1,pnts2)
imgOutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow('cards',imgOutput)
cv2.waitKey()