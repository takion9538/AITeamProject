import cv2
import sys
import numpy as np

src = cv2.imread('cat.bmp')

cp = (src.shape[1]/2, src.shape[0]/2)
rot = cv2.getRotationMatrix2D(cp, 20, 0.5) # 축소되는 원리 검색해보기
dst = cv2.warpAffine(src, rot, (0, 0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()