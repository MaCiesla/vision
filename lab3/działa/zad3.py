import cv2

import numpy as np



TP = 0
FP = 0
FN = 0
I0 = cv2.imread('input/in%06d.jpg' % 1)
I0G = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
MOG = cv2.createBackgroundSubtractorMOG2()



f =open('temporalROI.txt','r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start =int(roi_start)
roi_end =int(roi_end)
f.close()

kernel = np.ones((3,3),np.uint8)


for i in range(200,1020):
    I = cv2.imread('input/in%06d.jpg' % i)
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    model = MOG.apply(IG)
    # diff = cv2.absdiff(IG,model)
    B = cv2.threshold(model, 40, 255, cv2.THRESH_BINARY)
    B =B[1]
    median = cv2.medianBlur(B, 5)
    Dil = cv2.dilate(median,kernel,iterations = 1)
    Eros = cv2.erode(Dil, kernel, iterations=1)
    ER = Eros
    I0G = IG
        #indeksacja
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(Eros)
    img = np.uint8(labels / stats.shape[0] * 255)

    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        tab = stats[1:, 4]
        pi = np.argmax(tab)
        pi = pi + 1
        # wyrysownie box
        cv2.rectangle(I, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]),(255, 0, 0), 2)
        cv2.putText(I, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(I, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.imshow("eros", Eros)
    cv2.imshow("aproksymacja sredniej konserwatywnie", model)
    cv2.imshow("image", I)
    cv2.waitKey(10)

    if (i >= roi_start and i <= roi_end):
        I00G = cv2.imread('groundtruth/gt%06d.png' % i)
        B0 = cv2.threshold(I00G, 20, 255, cv2.THRESH_BINARY)
        B0 =B0[1]
        B0 = B0[:,:,0]


        TP_M = np.logical_and((Eros == 255),(B0 == 255))
        TP_S = np.sum(TP_M)
        TP = TP + TP_S

        FP_M = np.logical_and((Eros == 255),(B0 == 0))
        FP_S = np.sum(FP_M)
        FP = FP + FP_S

        FN_M = np.logical_and((Eros == 0),(B0 == 255))
        FN_S = np.sum(FN_M)
        FN = FN + FN_S




P = np.float32(TP)/(TP + FP)
R = np.float32(TP)/(TP + FN)
F1 = 2*P*R/(P+R)
print("P = " + str(P))
print("R = " + str(R))
print("F1 = " + str(F1))