#-*- coding:utf-8 -*-
import cv2
import numpy as np

def crop_minAreaRect(img, rect):
    #def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center

    #center = rect[0]

    #mult = 1.2
    size = rect[1]
    angle = rect[2]
    (o_h, o_w) = size[0], size[1]
    (cX, cY) = rect[0]
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotated = False
    angle = rect[2]



    #if (angle < -45 and (h > w)) or angle == -90 :
    if (angle < -45 ) or angle == -90:
        angle += 90
        rotated = True
    w = o_h if rotated==False else o_w
    h = o_w if rotated==False else o_h

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    img_rot = cv2.warpAffine(img, M, (nW, nH))
    #img_crop = cv2.getRectSubPix(img_rot, (int(w*0.9), int(h*0.9)),    (nW/2, nH/2))


    # r_w = h if rotated==False else w
    # r_h = w if rotated==False else h

    # r_w = w if rotated==False else h
    # r_h = h if rotated==False else w

    #img_crop = cv2.getRectSubPix(img_rot, (int(r_w * 0.9), int(r_h * 0.9)), (nW / 2, nH / 2))
    img_crop = cv2.getRectSubPix(img_rot, (int(w * 0.9), int(h * 0.9)), (nW / 2, nH / 2))
    if abs(rect[2]) < 3.0:
        img_crop = cv2.getRectSubPix(img, (int(w), int(h)), (cX, cY))


    if img_crop[:, :, 2].mean() > 170: #or angle == 0.0:
        img_crop = img

    return img_crop


csv_path ='../MURA-v1.1/train_image_paths.csv'
root = "../"
part = 'XR_ELBOW'
with open(csv_path, 'rb') as F:
    d = F.readlines()
    imgs = [root + str(x, encoding='utf-8').strip() for x in d if
            str(x, encoding='utf-8').strip().split('/')[2] == part]

# read pandas
#check_file_path = []
#
# imgs = [
# '../MURA-v1.1/train/XR_ELBOW/patient04903/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04905/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04910/study1_positive/image1.png' # 11
# ,'../MURA-v1.1/train/XR_ELBOW/patient04357/study1_positive/image1.png' #-51 #1
# ,'../MURA-v1.1/train/XR_ELBOW/patient04907/study1_positive/image2.png' # 몬가이상
# ,'../MURA-v1.1/train/XR_ELBOW/patient02188/study1_positive/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04923/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient03107/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient03260/study1_positive/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient05069/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient05063/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient05027/study1_positive/image1.png'
# , '../MURA-v1.1/train/XR_ELBOW/patient05025/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04971/study1_positive/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient00894/study2_negative/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04938/study1_positive/image1.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient04930/study1_positive/image2.png'
# ,'../MURA-v1.1/train/XR_ELBOW/patient03260/study1_positive/image2.png'
# ,'../MURA-v1.1/valid/XR_ELBOW/patient11389/study1_positive/image2.png'
# ,'../MURA-v1.1/valid/XR_ELBOW/patient11389/study1_positive/image4.png'
# ,'../MURA-v1.1/valid/XR_ELBOW/patient11390/study2_negative/image3.png'
# ]

for img_path in imgs:
    #refFilename = '../MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png'  # imgs[0]#
    #refFilename = '../MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04903/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04905/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04910/study1_positive/image1.png' # 11
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04357/study1_positive/image1.png' #-51 #1
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04907/study1_positive/image2.png' # 몬가이상
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient02188/study1_positive/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04923/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient03107/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient03260/study1_positive/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient05069/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient05063/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient05027/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient05025/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04971/study1_positive/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient00894/study2_negative/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04938/study1_positive/image1.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient04930/study1_positive/image2.png'
    #img_path = '../MURA-v1.1/train/XR_ELBOW/patient03260/study1_positive/image2.png'
    #img_path = '../MURA-v1.1/valid/XR_ELBOW/patient11389/study1_positive/image2.png'

    #img_path = '../MURA-v1.1/valid/XR_ELBOW/patient11389/study1_positive/image4.png'
    #img_path = '../MURA-v1.1/valid/XR_ELBOW/patient11390/study2_negative/image3.png'

    img = cv2.imread(img_path)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(imgray,(5,5),2)

    ret, thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,301, 0)
    #contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
    #hierachy는 contours line의 계층 구조
    #image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE  ,cv2.CHAIN_APPROX_SIMPLE)
    # Defining a condition to draw the contours, a number larger than/smaller than or between ranges
    x = 750.0
    removed_contour = []
    for c in contours:
        print(cv2.contourArea(c))
        if cv2.contourArea(c) > x:
            print(cv2.contourArea(c))
            removed_contour.append(c)
            #contours.remove(c)
            #x, y, w, h = cv2.boundingRect(c)

    if len(removed_contour) ==0:
        removed_contour = contours

    img_croped = cv2.drawContours(img, contours, -1, (20, 255, 180), 3)
    cnt = np.concatenate(removed_contour)


    rotrect = cv2.minAreaRect(cnt)
    img_croped = cv2.drawContours(img_croped, removed_contour, -1, (0, 255, 0), 3)
    box = np.int0(cv2.boxPoints(rotrect))
    img_croped = cv2.drawContours(img_croped, [box], 0, (36, 255, 12), 3)  # OR
    ellipse = cv2.fitEllipse(cnt)
    #img_croped = cv2.ellipse(img_croped, ellipse, (0,255,0),3)

    (e_x,e_y),e_r = cv2.minEnclosingCircle(cnt)
    center = (int(e_x), int(e_y))
    img_croped = cv2.circle(img_croped, center, int(e_r), (255,0,0),3)

    # crop
    img_croped = crop_minAreaRect(img, rotrect)


    cv2.imshow("_".join(img_path.split('/')[4:]), img_croped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()