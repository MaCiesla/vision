import cv2

I = cv2.imread('mandril.jpg')
cv2.imshow("Mandril kolor",I)

cv2.imwrite("m.png",I)
I2 = cv2.imread("m.png")
cv2.imshow("png",I2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(I.dtype)
print(I.shape)
print(I.size)
print(I2.size)