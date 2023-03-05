import sys
import numpy as np
import cv2

src = cv2.imread('cat.bmp') # src.shape = (600, 600)

dst1 = cv2.resize(src, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
dst2 = cv2.resize(src, (2400, 2400)) # cv2.INTERLINEAR
dst3 = cv2.resize(src, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
dst4 = cv2.resize(src, (2400, 2400), interpolation=cv2.INTER_LANCZOS4)

cv2.imshow('dst1', dst1[700:1500, 600:1700])
cv2.imshow('dst2', dst2[700:1500, 600:1700])
cv2.imshow('dst3', dst3[700:1500, 600:1700])
cv2.imshow('dst4', dst4[700:1500, 600:1700])

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()