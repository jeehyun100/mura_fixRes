from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500

GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score

    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h


if __name__ == '__main__':
    # Read reference image

    csv_path ='../MURA-v1.1/train_image_paths.csv'
    root = "../"
    part = 'XR_ELBOW'
    with open(csv_path, 'rb') as F:
        d = F.readlines()
        imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                str(x, encoding='utf-8').strip().split('/')[2] == part]

    #'./MURA-v1.1/train/XR_ELBOW/patient04903/study1_positive/image1.png'
    #'./MURA-v1.1/train/XR_ELBOW/patient04903/study1_positive/image1.png'
    #'/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image2.png''
    #'/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png'
    #/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png
    refFilename = '/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png' #imgs[0]
    #refFilename2 = '/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png'

    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    #imFilename = "scanned-form.jpg"
    imFilename = '/home/data/extra/mura_fixRes/MURA-v1.1/train/XR_ELBOW/patient00031/study1_negative/image1.png'
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)

    #cv2.imshow('image', imReg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #while True:

    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
