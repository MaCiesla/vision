import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("I.jpg")
J = cv2.imread("J.jpg")

IG = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
JG = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

diff = cv2.absdiff(IG,JG)

dX = dY = 1
W2=1
u = np.zeros(JG.shape)
v = np.zeros(IG.shape)

dis = []

for j in range(2,I.shape[0]-2):
    for i in range(2,I.shape[1]-2):
        IO = np.float32(IG[j-W2:j+W2+1,i-W2:i+W2+1])
        # print("dziala")
        min_dis = 99999999
        for jj in range(-dX,dX+1):
            for ii in range(-dY,dY+1):
                JO = np.float32(JG[j+jj-W2:j+jj+W2+1, i+ii-W2:i+ii+W2+1])
                dis = np.sum(np.sqrt((np.square(JO - IO))))
                if(dis < min_dis):
                    min_dis = dis
                    wsp_dis = [jj,ii]
        u[j, i] = wsp_dis[0]
        v[j, i] = wsp_dis[1]

plt.figure()
plt.gca().invert_yaxis()
plt.quiver(u,v)
plt.show()
cv2.imshow("I",IG)
cv2.imshow("J",JG)
cv2.imshow("diff",diff)
cv2.waitKey(0)