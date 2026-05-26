# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:24:18 2026

@author: luisf
"""
import cv2 as cv

def main():
    video=cv.VideoCapture("ant_video.mp4")
    while(True):
        is_true,frame=video.read()
        if is_true:
            cv.imshow("Video",frame)
            if cv.waitKey(1) & 0xFF==ord('d'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()
        
main()

