import cv2
import matplotlib.pyplot as plt
import numpy as np



I = cv2.imread("mandril.jpg")
IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
IGE = cv2.equalizeHist(IG)
#
plt.figure(1)
# plt.hist(x)
# plt.figure(2)
plt.title("histogram")
plt.hist(IG.ravel(),256,[0,256])


plt.figure(2)
plt.hist(IGE.ravel(),256,[0,256])
plt.title("histogram wyrownany globalnie")
# hist = cv2.calcHist([IG],[0],None,[256],[0,256])
# print(hist)
# plt.hist(hist)

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
I_CLAHE = clahe.apply(IG)
plt.figure(3)
plt.hist(I_CLAHE.ravel(),256,[0,256])
plt.title("histogram wyrownany CLAHE")

plt.figure(4)
plt.imshow(IG)
plt.title("normalny")

plt.figure(5)
plt.imshow(IGE)
plt.title("globalny")

plt.figure(6)
plt.imshow(I_CLAHE)
plt.title("Clahe")


plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()