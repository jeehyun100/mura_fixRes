#-*- coding:utf-8 -*-
import cv2
import numpy as np

def crop_minAreaRect(img, rect):
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
4
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot
    # rotate img
    # angle = rect[2]
    # rows,cols = img.shape[0], img.shape[1]
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    # img_rot = cv2.warpAffine(img,M,(cols,rows))
    #
    # # rotate bounding box
    # rect0 = (rect[0], rect[1], 0.0)
    # box = cv2.boxPoints(rect0)
    # pts = np.int0(cv2.transform(np.array([box]), M))[0]
    # pts[pts < 0] = 0
    #
    # # crop
    # img_crop = img_rot[pts[1][1]:pts[0][1],
    #                    pts[1][0]:pts[2][0]]

    #return img_crop

refFilename = '/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png'  # imgs[0]#
#refFilename = '/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image2.png'
img = cv2.imread(refFilename)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#threshold를 이용하여 binary image로 변환
#ret, thresh = cv2.threshold(imgray,109, 111, cv2.THRESH_BINARY)
# Otsu's thresholding
#ret,thresh = cv2.threshold(imgray,60,70,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret,thresh = cv2.threshold(imgray,0,135,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(imgray,(5,5),0)

ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
#hierachy는 contours line의 계층 구조
#image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL  ,cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
#
# maxsize = 0
# best = 0
# count = 0
# for cnt in contours:
#     if cv2.contourArea(cnt) > maxsize:
#         maxsize = cv2.contourArea(cnt)
#         best = count
#     count += 1
#
# #cv2.drawContours(img_rgb, contours[best], -1, (0,0,255), 2)
# contours2 = np.array(contours).reshape((-1,1,2)).astype(np.int32)
#image = cv2.drawContours(img, cnt, -1, (0,255,0), 1)
rotrect = cv2.minAreaRect(cnt)


# crop
img_croped,_ = crop_minAreaRect(img, rotrect)



# box = cv2.boxPoints(rotrect)
# box = np.int0(box)
# image = cv2.drawContours(img, [box], 0, (0,0,255), 2)

cv2.imshow('image', img_croped)
cv2.waitKey(0)
cv2.destroyAllWindows()