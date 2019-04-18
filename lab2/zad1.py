import cv2
import matplotlib.pyplot as plt
import numpy as np


TP = 0
I0 = cv2.imread('input/in%06d.jpg' % 1)
I0G = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)

B = cv2.threshold(I0G, 180, 255, cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
# cv2.imshow("I",B[1])
# cv2.waitKey(0)
# print(B[1])

for i in range(200,1020):
    I = cv2.imread('input/in%06d.jpg' % i)
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(IG,I0G)
    B = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    B =B[1]
    median = cv2.medianBlur(B, 5)
    Eros = cv2.erode(median,kernel,iterations = 1)
    Dil = cv2.dilate(Eros, kernel, iterations=1)
    # cv2.imshow("Dil",Dil)
    # cv2.waitKey(10)
    I0G = IG
        #indeksacja
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(Dil)
    img = np.uint8(labels / stats.shape[0] * 255)

    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        tab = stats[1:, 4]  # wyciecie 4 kolumny bez pierwszego elementu
        pi = np.argmax(tab)  # znalezienie indeksu najwiekszego elementu
        pi = pi + 1  # inkrementacja bo chcemy indeks w stats, a nie w tab
        # wyrysownie bbox
        cv2.rectangle(I, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]),(255, 0, 0), 2)
        cv2.putText(I, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(I, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.imshow("Labels", I)
    cv2.waitKey(10)

    I00G = cv2.imread('groundtruth/gt%06d.png' % i)
    B0 = cv2.threshold(I00G, 20, 255, cv2.THRESH_BINARY)
    B0 =B0[1]
    B0 = B0[:,:,0]
    # cv2.imshow("Labels", B0)
    # cv2.waitKey(10)

    TP_M = np.logical_and((Dil == 255),(B0 == 255)) # iloczyn logiczny odpowiednich elementow macierzy
    TP_S = np.sum(TP_M)        # suma elementow w macierzy
    TP = TP + TP_S             # aktualizacja wskaznika globalnego


f =open('temporalROI.txt','r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start =int(roi_start)
roi_end =int(roi_end)
print(TP)