import cv2
import numpy as np
import matplotlib.pyplot as plt

def rmsdiff(im1, im2):
    diff = np.abs(im1 - im2)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            diff[i, j] = diff[i, j]**2
    sum = np.sum(diff)
    N = im1.shape[0] * im1.shape[1]
    return np.sqrt(sum/N)

def bad20(im1, im2):
    cnt = 0
    diff = np.abs(im1 - im2)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i,j] > 2:
                cnt += 1
    N = im1.shape[0] * im1.shape[1]
    return (cnt/N)*100



imgL = cv2.pyrDown( cv2.imread('aloes/aloeL.jpg') )  # downscale images for faster processing
imgR = cv2.pyrDown( cv2.imread('aloes/aloeR.jpg') )
disp1 = cv2.pyrDown( cv2.imread('disp1.jpg',0) )
# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)


disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
disp = (disp-min_disp)/num_disp
# disp = np.uint8((disp - disp.min())/disp.max() * 255)
cv2.imshow('model disparity', disp1)
cv2.imshow('disparity', disp)
cv2.waitKey()
cv2.destroyAllWindows()
print("RMS: ",rmsdiff(disp,disp1))
print("Bad 2.0: ",bad20(disp, disp1))
