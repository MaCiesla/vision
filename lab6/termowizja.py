#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:35:15 2019

@author: student
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.ones((3, 3), np.uint8)
cap = cv2.VideoCapture('vid1_IR.avi')
iPedestrian = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, B = cv2.threshold(G, 45, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(B, 5)
    Dil = cv2.dilate(median, kernel, iterations=1)
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
    stats = np.matrix(output[2])
    # The fourth cell is the centroid matrix
    centroids = np.matrix(output[3])
    stats1 = np.zeros((1, 5))
    stats2 = np.zeros((1, 5))
    stats3 = np.zeros((1, 5))
    centroids1 = np.zeros((1, 2))
    for i in range(stats.shape[0]):
        if stats[i,4] > 900 and stats[i,4] < 10000:
            stats1 = np.append(stats1, stats[i, :], axis=0)
            centroids1 = np.append(centroids1, centroids[i, :], axis=0)
    stats = np.matrix(stats1.astype(int))
    centroids = np.matrix(centroids1.astype(int))
    if stats.shape[0] > 1:
        for i in range(stats.shape[0]):
            if stats[i, 4] > 0:
                stats2 = np.append(stats2, stats[i, :], axis=0)
                for j in range(stats.shape[0]):
                    if centroids[j, 0] > stats[i, 0] and centroids[j, 0] < stats[i, 0] + stats[i, 2] and i != j:
                        tmp = stats[i, :]
                        tmp[0,0] = np.min([stats[i, 0], stats[j, 0]])
                        tmp[0,1] = np.min([stats[i, 1], stats[j, 1]])
                        tmp[0,2] = np.max([stats[i, 2], stats[j, 2]])
                        tmp[0,3] = np.sum([stats[i, 3], stats[j, 3]])
                        tmp[0,4] = np.sum([stats[i, 4], stats[j, 4]])
                        stats2[i, :] = tmp
                        stats[j, :] = np.zeros((1, 5))


    stats = np.matrix(stats2.astype(int))

    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        for i in range(stats.shape[0]):
            tab = stats[1:, 4]
            pi = np.argmax(tab)
            pi = pi + 1
            a = np.abs(stats[i,2] - stats[i,0])
            b = np.abs(stats[i,3] - stats[i,1])
            size = a * b
            ind = np.zeros(num_labels)
            cv2.rectangle(G, (stats[i,0], stats[i,1]), (stats[i,0] + stats[i,2], stats[i,1] + stats[i,3]), (255, 255, 0), 2)
                #                cv2.putText(G, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                #                cv2.putText(G, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            ROI = G[stats[i,1]:stats[i,1] + stats[i,3], stats[i,0]:stats[i,0] + stats[i,2]]
            # cv2.imwrite('data1/sample_%06d.png' % iPedestrian, ROI)
            iPedestrian += 1

    cv2.imshow('IR', G)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()



