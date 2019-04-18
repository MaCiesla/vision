import cv2
import matplotlib.pyplot as plt

I = cv2.imread("mandril.jpg")
height, width = I.shape[:2]

scale = 2
Ix2 = cv2.resize(I,(int(scale * height),int(scale * width)))
cv2.imshow("dupaI",Ix2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(Ix2.dtype)