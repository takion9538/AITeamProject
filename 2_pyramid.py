import sys
import cv2
import numpy as np

src = cv2.imread('cat.bmp')

rc = (180, 180, 200, 200)

cpy = src.copy()
cv2.rectangle(cpy, rc, (0, 0, 255), 2) # Red, 두께 :2
cv2.imshow('src', cpy)
cv2.waitKey()
cv2.destroyAllWindows()

for i in range(1, 4) :
    src = cv2.pyrDown(src)
    cpy = src.copy()
    cv2.rectangle(cpy, rc, (0, 0, 255), 2, shift=i) # shift : 해당 배수만큼 줄어든다. (1/i)
    cv2.imshow('src', cpy)
    cv2.waitKey()
    cv2.destroyWindow('src')

cv2.destroyAllWindows()