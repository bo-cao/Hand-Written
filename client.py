import torch
import image
import numpy as np
import cv2
IMAGE_PATH = "00.jpg"
im = cv2.imread(IMAGE_PATH)
img_hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
lower= np.array([0, 0, 0])
upper = np.array([180, 255, 100])
mask0 = cv2.inRange(img_hsv,lower,upper)
cv2.imshow('b',mask0)
cv2.waitKey(1000)
# contours , hierarchy = cv2.findContours(mask0 , cv2.RETR_LIST ,
#         cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(mask0)
mask0 = mask0[max(0,y-20):min(y+h+20,mask0.shape[0]),max(0,x-20):min(x+w+20,mask0.shape[1])]
mask = 255 - mask0
cv2.imshow('a',mask)
cv2.waitKey(1000)
cv2.imwrite("convert_image.jpg",mask)
model = torch.load("./model.pkl")
ans, pred = image.predict(model=model, image=image.get_image(image_path="convert_image.jpg"))
print(ans)
