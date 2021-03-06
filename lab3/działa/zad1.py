import cv2
import matplotlib.pyplot as plt
import numpy as np


TP = 0
FP = 0
FN = 0
I0 = cv2.imread('input/in%06d.jpg' % 1)
I0G = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)



N = 60
YY, XX = I0G.shape
BUF = np.zeros((YY, XX, N), np.uint8)
BUF1 = np.zeros((YY, XX, N), np.uint8)
iN = 0
iN1 = 0

def mean_buf(IG):
    global iN
    BUF [:,:,iN] = IG
    iN = iN + 1
    if iN == N:
        iN = 0
    mean = np.mean(BUF,axis = 2)
    return np.uint8(mean)

def median_buf(IG):
    global iN1
    BUF1 [:,:,iN1] = IG
    iN1 = iN1 + 1
    if iN1 == N:
        iN1 = 0
    median = np.median(BUF1,axis = 2)
    return np.uint8(median)

BG = np.zeros((YY, XX), np.uint8)
BG1 = np.zeros((YY, XX), np.uint8)
ER = np.zeros((YY, XX), np.uint8)
BG2 = np.zeros((YY, XX), np.uint8)
BG2 = I0G
alfa = 0.05

def aprok_mean(IG):
    global BG
    BGn =np.float64(alfa*np.float64(IG) + (1-alfa)*np.float64(BG))
    BG = BGn
    return np.uint8(BGn)

def aprok_median(IG):
    global BG1
    BGn = np.float64((BG1) + np.sign(IG-BG1))
    BG1 = BGn
    return np.uint8(BGn)

def aprok_mean_con(IG):
    global ER
    global BG2
    BGn = np.zeros((YY, XX), np.uint8)
    for i in range(YY):
        for j in range(XX):
            if ER[i,j] == 0:
                BGn[i,j] = np.float64(alfa*np.float64(IG[i,j]) + (1-alfa)*np.float64(BG2[i,j]))
            else:
                BGn[i,j] = IG[i,j]
    BG2 = BGn
    return np.uint8(BGn)





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
    model_meanbuf = mean_buf(IG)
    model_medianbuf = median_buf(IG)
    model_meanapro = aprok_mean(IG)
    model = aprok_median(IG)
    diff = cv2.absdiff(IG,model)
    B = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
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
    cv2.imshow("bufor srednia", model_meanbuf)
    cv2.imshow("bufor mediana", model_medianbuf)
    cv2.imshow("aproksymacja sredniej", model_meanapro)
    cv2.imshow("aproksymacja mediany", model)
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