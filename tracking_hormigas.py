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

def individualizar_hormigas(coordenadas,e,p):
    
    #Individualizacion con DBSCAN
    clustering = DBSCAN(eps=e, min_samples=p).fit(coordenadas)
    
    #Clusters  correspondiente a cada punto y ruido
    labels=clustering.labels_
    
    #Depuracion de clusters que probablemente son ruido
    unicos_labels, cantidad = np.unique(labels, return_counts=True)
    if len(unicos_labels)!=1: 
        promedio_puntos_por_cluster=((len(labels)-(np.count_nonzero(labels == -1)))/(len(unicos_labels)-1))*0.55
        es_pequeño= (cantidad < promedio_puntos_por_cluster ) & (unicos_labels != -1)
        labels_a_eliminar = unicos_labels[es_pequeño]
        labels[np.isin(labels, labels_a_eliminar)] = -1
    return labels


def coordenadas_en_los_extremos_de_cluster(labels,coordenadas_x,coordenadas_y,l):
    
    #Coordenadas x ,  y de un especifico label
    x_de_l=coordenadas_x[labels==l]
    y_de_l=coordenadas_y[labels==l]
    return  min(x_de_l),max(x_de_l),min(y_de_l),max(y_de_l)

def aplicar_mascara_frame(frame,coordenadas):
    h, w = frame.shape[:2]
    
    #Definicion de la mascara negra
    mask = np.zeros((h, w), dtype=np.uint8)
    
    #Cracion de rectangulo blanco con las cordenadas del area visible
    cv.rectangle(mask, coordenadas[0], coordenadas[1], 255, thickness=-1)
    
    #Aplicacion de mascara en el frame
    return cv.bitwise_and(frame, frame, mask=mask)

def punto_central_frame(labels,coordenadas_x,coordenadas_y,l):
    #Coordenadas x ,  y de un especifico label
    x_de_l=coordenadas_x[labels==l]
    y_de_l=coordenadas_y[labels==l]
    
    #Calcula el promedio de puntos ,que es equivalente al punto central
    return  (int(sum(x_de_l)/len(x_de_l)),int(sum(y_de_l)/len(x_de_l)))
    
    
#Lectura de video de hormigas y despliegue de video procesado

def main():
    #Lectura primer frame
    video=cv.VideoCapture("ant_video.mp4")
    is_true,frame=video.read()
    
    #Deteccion de hormigas
    coordenadas_posibles_hormigas=detectar_posibles_hormigas(frame)
    labels_hormigas=individualizar_hormigas(coordenadas_posibles_hormigas,10,50)
    
    
    #Dibujo de rectangulos de deteccion de hormigas
    coordenadas_x=coordenadas_posibles_hormigas[:,0]
    coordenadas_y=coordenadas_posibles_hormigas[:,1]
    coordenadas_bordes_rectangulos=[]
    lista_labels=np.unique(labels_hormigas).tolist()
    lista_labels.remove(-1)
    frame_con_rectangulos=frame.copy()
    for l in lista_labels:
        x_minimo_de_l,x_maximo_de_l,y_minimo_de_l,y_maximo_de_l=coordenadas_en_los_extremos_de_cluster(labels_hormigas, coordenadas_x, coordenadas_y, l)
        esquina_superior_izquierda=(x_minimo_de_l-10,y_minimo_de_l-10)
        esquina_inferior_derecha=(x_maximo_de_l+10,y_maximo_de_l+10)
        cv.rectangle(frame_con_rectangulos,esquina_superior_izquierda,esquina_inferior_derecha, (0, 255, 0), 2)
        cv.putText(frame_con_rectangulos,f"{l}",esquina_superior_izquierda, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        coordenadas_bordes_rectangulos.append([esquina_superior_izquierda, esquina_inferior_derecha])
    cv.imshow("Primer frame", frame_con_rectangulos)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
    #Seleccion de label a trackear 
    label_a_trackear=int(input(f"Selecciona uno de los siguientes labels para ver su recorrido: {lista_labels}  "))
    
    #Cordenadas de mascara inicial para enfocarse en el label a trackear
    coordenadas_rectangulo_label=coordenadas_bordes_rectangulos[lista_labels.index(label_a_trackear)]
    coordenadas_mascara=[(coordenadas_rectangulo_label[0][0]-10,coordenadas_rectangulo_label[0][1]-10),(coordenadas_rectangulo_label[1][0]+10,coordenadas_rectangulo_label[1][1]+10)]
    
    #Punto central inicial
    punto_central= punto_central_frame(labels_hormigas,coordenadas_x,coordenadas_y,label_a_trackear)
    
    coordendas_x_recorrido=[]
    coordenadas_y_recorrido=[]
    #Tracking de label seleccionado
    while(True):
        is_true,frame=video.read()
        if is_true:
            frame=aplicar_mascara_frame(frame, coordenadas_mascara)
            coordenadas_posibles_hormigas=detectar_posibles_hormigas(frame)
            labels_hormigas=individualizar_hormigas(coordenadas_posibles_hormigas,5,30)
            coordenadas_x=coordenadas_posibles_hormigas[:,0]
            coordenadas_y=coordenadas_posibles_hormigas[:,1]
            print(np.unique(labels_hormigas))
            filas=frame.shape[0]
            colours = [(0.5, 0.5, 0.5) if label == -1 else (1, 0, 0) for label in labels_hormigas]
            plt.scatter(coordenadas_x,filas-coordenadas_y,c=colours,s=1)
            plt.show()
            cv.imshow("Video",frame)
            if cv.waitKey(1) & 0xFF==ord('d'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()
        
main()

