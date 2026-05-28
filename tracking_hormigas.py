# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:24:18 2026

@author: luisf
"""
import cv2 as cv
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


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
    cond_thresh = r > 68
    cond_oscuro_r = r < 140
    cond_oscuro_g = g < 90
    cond_oscuro_b = b < 90
    
    #Combinacion de condiciones 
    mask_hormigas = cond_green & cond_blue & cond_thresh & cond_oscuro_r & cond_oscuro_g & cond_oscuro_b
    y_indices, x_indices = np.where(mask_hormigas)
    for x, y in zip(x_indices, y_indices):
        coordenadas.append([x,y])
        
    return np.array(coordenadas)
    #return frame

def individualizar_hormigas(coordenadas):
    
    #Individualizacion con DBSCAN
    clustering = DBSCAN(eps=10, min_samples=50).fit(coordenadas)
    
    #Clusters  correspondiente a cada punto y ruido
    labels=clustering.labels_
    
    #Depuracion de clusters que probablemente son ruido
    promedio_puntos_por_cluster=((len(labels)-(np.count_nonzero(labels == -1)))/(len(np.unique(labels))-1))*0.55
    unicos_labels, cantidad = np.unique(labels, return_counts=True)
    es_pequeño= (cantidad < promedio_puntos_por_cluster ) & (unicos_labels != -1)
    labels_a_eliminar = unicos_labels[es_pequeño]
    labels[np.isin(labels, labels_a_eliminar)] = -1

    return labels


def coordenadas_en_los_extremos_de_cluster(labels,coordenadas_x,coordenadas_y,l):
    
    #Coordenadas x ,  y de un especifico label
    x_de_l=coordenadas_x[labels==l]
    y_de_l=coordenadas_y[labels==l]
    return  min(x_de_l),max(x_de_l),min(y_de_l),max(y_de_l)

#Lectura de video de hormigas y despliegue de video procesado

def main():
    video=cv.VideoCapture("ant_video.mp4")
    is_true,frame=video.read()
    coordenadas_posibles_hormigas=detectar_posibles_hormigas(frame)
    coordenadas_x=coordenadas_posibles_hormigas[:,0]
    coordenadas_y=coordenadas_posibles_hormigas[:,1]
    labels_hormigas=individualizar_hormigas(coordenadas_posibles_hormigas)
    
    #Dibujo de areas de deteccion de hormigas
    for l in np.unique(labels_hormigas):
        if l!=-1:
            x_minimo_de_l,x_maximo_de_l,y_minimo_de_l,y_maximo_de_l=coordenadas_en_los_extremos_de_cluster(labels_hormigas, coordenadas_x, coordenadas_y, l)
            cv.rectangle(frame,(x_minimo_de_l-10,y_minimo_de_l-10),(x_maximo_de_l+10,y_maximo_de_l+10), (0, 255, 0), 2)
            cv.putText(frame,f"{l}",(x_minimo_de_l-10,y_minimo_de_l-16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.imshow("Primer frame", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    label_a_trackear=input(f"Selecciona uno de los siguientes labels para ver su recorrido: {np.unique(labels_hormigas)}")

    
    
    # coordenadas_x=coordenadas_posibles_hormigas[:,0]
    # coordenadas_y=coordenadas_posibles_hormigas[:,1]
    # colours=individualizar_hormigas(coordenadas_posibles_hormigas)
    # plt.scatter(coordenadas_x,coordenadas_y,c=colours,s=1)
    
    
    # while(True):
    #     is_true,frame=video.read()
    #     if is_true:
    #         #frame=detectar_posibles_hormigas(frame)
    #         coordenadas_posibles_hormigas=detectar_posibles_hormigas(frame)
            
    #         #Presentacion de hormigas individualizadas 

    #         coordenadas_x=coordenadas_posibles_hormigas[:,0]
    #         coordenadas_y=coordenadas_posibles_hormigas[:,1]
    #         colours=individualizar_hormigas(coordenadas_posibles_hormigas)
    #         plt.scatter(coordenadas_x,coordenadas_y,c=colours,s=1)
    #         plt.show()
    #         cv.imshow("Video",frame)
    #         if cv.waitKey(1) & 0xFF==ord('d'):
    #             break
    #     else:
    #         break
    # video.release()
    # cv.destroyAllWindows()
        
main()

