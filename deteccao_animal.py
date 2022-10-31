
import numpy as np
import cv2
import sys
from random import randint


video_source = 'Animal_1.mp4'
text_color = (0, 255, 0)
tracker_color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
bgs_types = ['GMG','MOG','MOG2','KNN','CNT']

def getkernel(kernel_type):
    if kernel_type == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if kernel_type == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if kernel_type == 'closing':
        kernel = np.ones((3,3), np.uint8)

    return kernel

def getfilter(img,filter):
    if filter == 'dilation':
        return cv2.dilate(img, getkernel('dilation'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getkernel('opening'), iterations=2)
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getkernel('closing'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getkernel('closing'), iterations=2) 
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getkernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, getkernel('dilation'), iterations=2)
        return dilation

def getsubtractor(bgs_type):
    if bgs_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if bgs_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if bgs_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if bgs_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if bgs_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()

    print('filtro invalido')
    sys.exit(1)


cap = cv2.VideoCapture(video_source)
minarea = 250
bg_subtractor = getsubtractor(bgs_types[2])
bgs_type = bgs_types[2]

def main():
    while cap.isOpened():
        ok, frame = cap.read()

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getfilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minarea:
                x, y, w, h, = cv2.boundingRect(cnt)
                
                #cv2.rectangle(frame, (20,30), (250,55),(0,0,255), -1)
                #cv2.putText(frame, 'Movimento detectado!', (10,50), font, 0.8, text_color, 2, cv2.LINE_AA)

                cv2.drawContours(frame, cnt, -1, tracker_color, 3)
                cv2.drawContours(frame, cnt, -1, (255,255,255), 1)
                cv2.rectangle(frame, (x,y), (x+w, y+h), tracker_color,3)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 1)

              

        result = cv2.bitwise_and(frame, frame, mask=bg_mask)

        if not ok:
            print('error')
            break

        #if bgs_type != 'MOG' and bgs_type !='GMG':
            #cv2.imshow('oi', bg_subtractor.getBackgroundImage())

        cv2.imshow('animals', result)
        cv2.imshow('frame',frame)

       
        
        
       



        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

main()
