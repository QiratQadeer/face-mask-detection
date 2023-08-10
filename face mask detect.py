# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 10:56:12 2023

@author: Qirat Qadeer
"""

from tensorflow.keras.models import load_model

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h,w)= frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
    (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    faces = []
    locs = []
    preds =[]
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1 endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvcolor(face, cv2.COLOR_BAYER_BG2BGR)
            face = cv2. resize(face, (244, 244))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size-32)
            return(locs, preds)
        prototxPath = r"face-detector\deploy.prototxT"
        weightsPath = r"face-detector\"
        