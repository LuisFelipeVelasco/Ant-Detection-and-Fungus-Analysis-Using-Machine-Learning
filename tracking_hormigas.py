# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:24:18 2026

@author: luisf
"""
import cv2 as cv
import numpy as np


def detectar_posibles_hormigas(frame):
    coordenadas=[]
    # frame[:,:,0] es blue, frame[:,:,1] es green, frame[:,:,2] es Red
    
    #Se extraen los canales y se pasan a int32 para prevenir errores matematicos
    b = frame[:, :, 0].astype(np.int32)
    g = frame[:, :, 1].astype(np.int32)
    r = frame[:, :, 2].astype(np.int32)
    
    # Condiciones para que un frame sea considerado como posible parte de hormiga
    cond_green  = r > (1.35 * g)
    cond_blue   = r > (1.35 * b)
    cond_thresh = r > 45
    
    #Combinacion de condiciones 
    mask_hormigas = cond_green & cond_blue & cond_thresh
    filas=frame.shape[1]
    y_indices, x_indices = np.where(mask_hormigas)
    for x, y in zip(x_indices, y_indices):
        coordenadas.append([x, filas - y])
        
    return coordenadas
    

#Lectura de video de hormigas y despliegue de video procesado

def main():
    video=cv.VideoCapture("ant_video.mp4")
    coordenadas_posibles_hormigas=[]
    while(True):
        is_true,frame=video.read()
        if is_true:
            coordenadas_posibles_hormigas=detectar_posibles_hormigas(frame)
            cv.imshow("Video",frame)
            if cv.waitKey(1) & 0xFF==ord('d'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()
        
main()

