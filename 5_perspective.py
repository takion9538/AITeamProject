import sys
import cv2
import numpy as np

src = cv2.imread('scanned.jpg')

h, w = src.shape[:2]
dw = 500
# A4 용지의 크기 : 210x297cm
dh = round(dw * 297/210)

srcQuad = np.array([[30, 30],
                    [30, 30],
                    [30, 30],
                    [30, 30]], np.float32)

dstQuad = np.array([[0, 0],
                    [w-1, 0],
                    [w-1, h-1],
                    [0, h-1]], np.float32)

dragSrc = [False, False, False, False]


def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:  # 원의 반지름인 25 = 원 크기 만큼만 상호작용되도록 설정
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break


def drawROI(img, corners):
    cpy = img.copy()
    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)  # -1 = 배경색 채우기

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    # 투명도 조절 옵션
    disp = cv2.addWeighted(img, -.3, cpy, 0.7, 0)

    return disp

disp = drawROI(src, srcQuad)

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
# pers : 3 x 3 Matrix 의 투시 변환 행렬
dst = cv2.warpPerspective(src, pers, (w, h))
# dst : pres 를 토대로 변환된 영상

cv2.imshow('img', disp)
cv2.setMouseCallback('img', onMouse)

while True :
    key = cv2.waitKey()
    if key == 13 :
        break
    elif key == 27 :
        cv2.destroyWindow('img')
        sys.exit()

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()