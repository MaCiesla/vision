import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
#import scipy
import numpy as np

# I = cv2.imread('mandril.jpg')
# cv2.imshow("Mandril",I)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imwrite("m.png",I)
# print(I.shape)
# print(I.size)
# print(I.dtype)


I = plt.imread('mandril.jpg')
#plt.figure(1)
fig,ax = plt.subplots(1)
plt.imshow(I)
plt.title('Mandril')
plt.axis('off')
x = [ 100, 150, 200, 250]
y = [ 50, 100, 150, 200]
plt.plot(x,y,'r.',markersize=10)
#plt.show()
plt.imsave('man.png',I)


rect = Rectangle((50,50),50,100,fill=False, ec='r');
ax.add_patch(rect)
plt.show()

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# plt.figure(2)
# plt.imshow(IG)
# plt.show()
#
# plt.figure(3)
# plt.imshow(IHSV)
# plt.show()
#
#
# IH = IHSV[:,:,0];
# IS = IHSV[:,:,1];
# IV = IHSV[:,:,2];
#
# print(IH)
# print(IS)
# print(IV)
#
#
# def rgb2gray(I):
#     return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]
#
# IG2 = rgb2gray(I)
# plt.figure(4)
# plt.gray()
# plt.imshow(IG2)
# plt.show()
#
#
# IHSV2 = matplotlib.colors.rgb_to_hsv(I)
# plt.figure(5)
# plt.imshow(IHSV2)
# plt.show()
#
#
# height, width = I.shape[:2]
# scale = 1.75
# Ix2 = cv2.resize(I,(int(scale*height),int(scale*width)))
# cv2.imshow("Big_Mandril",Ix2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# I_2 = scipy.misc.imresize(I, 0.5)
#
# cv2.imshow("Big_scipy",I_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


L = plt.imread('lena.png')
LG = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
suma = LG + IG
plt.figure(6)
plt.gray()
plt.imshow(suma)
plt.show()


roznica = LG - IG
plt.figure(7)
plt.gray()
plt.imshow(roznica)
plt.show()



iloczyn = LG * IG
plt.figure(8)
plt.gray()
plt.imshow(iloczyn)
plt.show()


C = 0.5*LG+0.5*IG
plt.figure(9)
plt.gray()
# C = np.uint8(C)
plt.imshow(C)
plt.show()


ABS = abs(IG-LG)
ABS = np.uint8(ABS)
plt.figure(10)
plt.gray()
# C = np.uint8(C)
plt.imshow(ABS)
plt.show()
