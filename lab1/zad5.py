import cv2
import matplotlib.pyplot as plt

I = cv2.imread("mandril.jpg")
L = cv2.imread("lena.png")
I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
L = cv2.cvtColor(L,cv2.COLOR_BGR2GRAY)

S = I+L
R = I-L
M = I*L

cv2.imshow("suma",S)
cv2.waitKey(0)
cv2.imshow("roznica",R)
cv2.waitKey(0)
cv2.imshow("mnozenie",M)
cv2.waitKey(0)
cv2.destroyAllWindows()


