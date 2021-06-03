import cv2
import numpy as np
b = input("enter a filename:   ")
path = ("C:\\Users\\vineesha thoutam\\Downloads\\" + b )
img = cv2.imread(path)
img = cv2.resize(img, (900,900))
a = int(input("enter the value:  ")) 
if a*2 == 16:
    cv2.imshow('actualimage', img)
if a == 9:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayimage', gray_image)
if a == 0:
    B, G, R = cv2.split(img)
    cv2.imshow('spliting', img)
if a == 3:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_range = np.array([0,0,100])    
    upper_range = np.array([0,255,0])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cv2.imshow('ABC', mask)
if a == 4:
    lower_range = np.array([0,200,200])    
    upper_range = np.array([5,255,255])
    mask = cv2.inRange(img, lower_range, upper_range)
    result = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow('abc', result)
if a == 5:
    logo = (r'C:\Users\vineesha thoutam\Pictures\logo.jpg')
    logo = cv2.resize(logo, (900,900))
    res = cv2,addWeighted(img, 0.9, logo, 0.4, 0.0)
    cv2.imshow('logo', logo)
    cv2.imshow('result', res)
if a == 6:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayimage', gray_image)   
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale([gray_image, [scalefactor = 1.0, [minNeighbors=3]]])
    for x, y, w, h in detected_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv2.imshow('facedetect', img)
if a == 7:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imshow("Image", img)
    cv2.imshow("edges", edges)
    cv2.imshow("Cartoon", cartoon) 
cv2.waitKey(0)      