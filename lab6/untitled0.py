#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:35:15 2019

@author: student
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.ones((3,3),np.uint8)
cap = cv2.VideoCapture('vid1_IR.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, B = cv2.threshold(G,45,255,cv2.THRESH_BINARY)
    median = cv2.medianBlur(B, 5)
    Dil = cv2.dilate(median,kernel,iterations = 1)
    Eros = cv2.erode(Dil, kernel, iterations=1)
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(Eros, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    
    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        for i in range (num_labels):
            tab = stats[:, 4]
            pi = np.argmax(tab)
            pi = pi + 1
            a = np.abs(stats[i][2] - stats[i][0])
            b = np.abs(stats[i][3] - stats[i][1])
            size = a*b
            ind = np.zeros(num_labels)
            if tab[i] > 900 and tab[i] < 10000:
#                for j in range (num_labels):
#                    if centroids[j,0] >= stats[i][0] and centroids[j,0] <= stats[i][0] + stats[i][2]:
#                        cv2.rectangle(G, (stats[j][0], stats[j][1]), (stats[j][0] + stats[i][2],stats[j][1] + stats[i][3]),(255, 0, 0), 2)
#                        ind[j] = 1
#                    elif ind [i] == 0:
                cv2.rectangle(G, (stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]),(255, 0, 0), 2)
#                cv2.putText(G, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
#                cv2.putText(G, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    
    cv2.imshow('IR',G)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()




