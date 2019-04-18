import cv2
import numpy as np
import matplotlib.pyplot as plt


def of(I, J, u0, v0, W2=1, dY=1, dX=1):
    u = np.zeros(JG.shape)
    v = np.zeros(IG.shape)

    for j in range(2, I.shape[0] - 2):
        for i in range(2, I.shape[1] - 2):
            IO = np.float32(IG[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
            min_dis = 99999999
            for jj in range(-dX+int(u0[j,i]), dX+int(u0[j,i])+1):
                for ii in range(-dY+int(v0[j,i]), dY+1+int(v0[j,i])):
                    JO = np.float32(JG[j + jj - W2:j + jj + W2 + 1, i + ii - W2:i + ii + W2 + 1])
                    dis = np.sum(np.sqrt((np.square(JO - IO))))
                    if (dis < min_dis):
                        min_dis = dis
                        wsp_dis = [u0[j,i]+jj, v0[j,i]+ii]
            u[j, i] = wsp_dis[0]
            v[j, i] = wsp_dis[1]
    return[u,v]

def pyramid(im, max_scale):
    images=[im];
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0,0), fx=0.5, fy=0.5))
    return images

u0 = np.zeros(IP[-1].shape, np.float32)
v0 = np.zeros(JP[-1].shape, np.float32)


I = cv2.imread("I.jpg")
J = cv2.imread("J.jpg")

IG = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
JG = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

diff = cv2.absdiff(IG,JG)

dX = dY = 1
W2=1
u = np.zeros(JG.shape)
v = np.zeros(IG.shape)

data = of(IG, JG, u, v, W2=1, dY=1, dX=1)


plt.figure()
plt.gca().invert_yaxis()
plt.quiver(data[0],data[1])
plt.show()
cv2.imshow("I",IG)
cv2.imshow("J",JG)
cv2.imshow("diff",diff)
cv2.waitKey(0)