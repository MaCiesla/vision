import cv2
import matplotlib.pyplot as plt



I = cv2.imread("mandril.jpg")
IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
plt.gray()
plt.imshow(IG)
plt.figure(1)
plt.title("dupa")
plt.axis('Off')
plt.show()


print (IG.shape)
print (IG)


IH = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
plt.figure(2)
plt.imshow(IH)
plt.axis("OFF")
plt.title("HSV")
plt.show()

print(IH[:,:,1])
