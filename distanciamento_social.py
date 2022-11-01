
import numpy as np
import cv2 
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
video_source = 'Pedestrians_2.mp4'
tracker_color = (0,251,37)
bgs_types = ['GMG','MOG','MOG2','KNN','CNT']
warning_color = (0,0,255)

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
    
def getsubtractor(bgs_types):
    if bgs_types == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if bgs_types == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()       
    if bgs_types == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if bgs_types == 'KNN':
        return cv2.createBackgroundSubtractorKNN() 
    if bgs_types == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    
    print('filtro invalido')
    sys.exit(1)

cap = cv2.VideoCapture(video_source)
bg_subtractor = getsubtractor(bgs_types[4])
bgs_type = bgs_types[4]
minarea = 400
maxarea = 3000

def main():
    while cap.isOpened:
        ok, frame = cap.read()

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getfilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask,5)
        result = cv2.bitwise_and(frame, frame, mask= bg_mask)


        (contour, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area >= minarea:
                x,y,w,h = cv2.boundingRect(cnt)

                #cv2.drawContours(frame, cnt, -1, tracker_color, 10)
                #cv2.drawContours(frame, cnt, -1, (255,255,255), 1)
                cv2.rectangle(frame, (x,y),(x+w, y+h), tracker_color, 1)
                #cv2.rectangle(frame, (x,y),(x+w, y+h), (255,255,255), 1)

            if area >= maxarea:
                cv2.rectangle(frame, (x,y),(x+ 120, y - 13), (49,49,49), 1)
                cv2.putText(frame, 'Aviso Distanciamento', (x,y -2), font, 0.4, (255,255,255),1, cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w, y+h), warning_color, 1)

        if not ok:
            print('error')
            break
        result = cv2.bitwise_and(frame, frame, mask= bg_mask)
        cv2.imshow('Pedestres2', result)
        cv2.imshow('Pedestres', frame)
        


        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

main()