import cv2
import matplotlib.pyplot as plt


I = cv2.imread("lena.png")
IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
IGa = cv2.GaussianBlur(I,(5,5),0)
IM = cv2.medianBlur(I,5)
ISx = cv2.Sobel(IG,cv2.CV_64F,1,0,ksize = 5)
ISy = cv2.Sobel(IG,cv2.CV_64F,0,1,ksize = 5)
IL = cv2.Laplacian(IG,cv2.CV_64F,ksize = 5)
cv2.imshow("normalny",I)
cv2.imshow("gray",IG)
cv2.imshow("gaussian",IGa)
cv2.imshow("sobel x",ISx)
cv2.imshow("sobel y",ISy)
cv2.imshow("median",IM)
cv2.imshow("laplasjan",IL)
cv2.waitKey(0)
cv2.destroyAllWindows()















