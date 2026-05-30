# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:24:18 2026

@author: luisf
"""
import cv2 as cv
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def detectar_coordenadas_hormigas(frame):
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

def individualizar_hormigas(coordenadas,e,p,b=1,average=True):
    
    #Individualizacion con DBSCAN
    clustering = DBSCAN(eps=e, min_samples=p).fit(coordenadas)
    
    #Clusters  correspondiente a cada punto y ruido
    labels=clustering.labels_
    
    #Depuracion de clusters que probablemente son ruido
    unicos_labels, cantidad = np.unique(labels, return_counts=True)
    if len(unicos_labels)!=1 and average: 
        promedio_puntos_por_cluster=((len(labels)-(np.count_nonzero(labels == -1)))/(len(unicos_labels)-1))*b
        es_pequeño= (cantidad < promedio_puntos_por_cluster ) & (unicos_labels != -1)
        labels_a_eliminar = unicos_labels[es_pequeño]
        labels[np.isin(labels, labels_a_eliminar)] = -1
    return labels


def conseguir_coordenadas_en_los_extremos_de_cluster(labels,coordenadas_x,coordenadas_y,l):
    
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

def conseguir_punto_central_frame(labels,coordenadas_x,coordenadas_y,l):
    #Coordenadas x ,  y de un especifico label
    x_de_l=coordenadas_x[labels==l]
    y_de_l=coordenadas_y[labels==l]
    #Calcula el promedio de puntos ,que es equivalente al punto central
    return  (int(sum(x_de_l)/len(x_de_l)),int(sum(y_de_l)/len(x_de_l)))
    
    
def separacion_fondo(video,n):
    #Escoge al azar  n ids entre 1 el numero total de  frames del video
    frame_ids = video.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=n)

    #Guarda los frames de los ids escogidos
    frames = []
    for fid in frame_ids:
        video.set(cv.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    # retorna un numpy array con la media de cada pixel de los n frames
    return np.median(frames, axis=0).astype(dtype=np.uint8)

def detectar_coordenadas_puntos_en_hongo(imagen):
    coordenadas=[]
    # imagen[:,:,0] es blue, imagen[:,:,1] es green,  imagen[:,:,2] es Red
    #Se extraen los canales y se pasan a int32 para prevenir errores matematicos
    b = imagen[:, :, 0].astype(np.int32)
    g = imagen[:, :, 1].astype(np.int32)
    r = imagen[:, :, 2].astype(np.int32)
    
    # Condiciones para que un frame sea considerado como posible parte del hongo
    cond_green  = r > (1 * g)
    cond_blue   = r > (1 * b)
    cond_thresh = r > 40
    cond_green_blue= g>b*1.03
    
    #Combinacion de condiciones 
    mask_hormigas = cond_green & cond_blue & cond_thresh & cond_green_blue
    y_indices, x_indices = np.where(mask_hormigas)
    for x, y in zip(x_indices, y_indices):
        coordenadas.append([x,y])     
    return np.array(coordenadas)

def main():
    #Lectura primer frame
    video=cv.VideoCapture("ant_video.mp4")
    is_true,frame=video.read()
    
    #Deteccion de hormigas
    coordenadas_hormigas=detectar_coordenadas_hormigas(frame)
    labels_hormigas=individualizar_hormigas(coordenadas_hormigas,10,50,b=0.55)
    
    #Dibujo de rectangulos de deteccion de hormigas
    coordenadas_x=coordenadas_hormigas[:,0]
    coordenadas_y=coordenadas_hormigas[:,1]
    coordenadas_bordes_rectangulos=[]
    lista_labels=np.unique(labels_hormigas).tolist()
    lista_labels.remove(-1)
    frame_con_rectangulos=frame.copy()
    for l in lista_labels:
        x_minimo_de_l,x_maximo_de_l,y_minimo_de_l,y_maximo_de_l=conseguir_coordenadas_en_los_extremos_de_cluster(labels_hormigas, coordenadas_x, coordenadas_y, l)
        esquina_superior_izquierda=(x_minimo_de_l,y_minimo_de_l)
        esquina_inferior_derecha=(x_maximo_de_l,y_maximo_de_l)
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
    coordenadas_mascara=[(coordenadas_rectangulo_label[0][0]-0,coordenadas_rectangulo_label[0][1]-0),(coordenadas_rectangulo_label[1][0]+0,coordenadas_rectangulo_label[1][1]+0)]
    punto_central=conseguir_punto_central_frame(labels_hormigas,coordenadas_x,coordenadas_y,label_a_trackear)
    
    coordenadas_x_recorrido=[punto_central[0]]
    coordenadas_y_recorrido=[punto_central[1]]
    
    deteccion_hongo=input("¿Desea saber en que parte del recorrido estuvo sobre el hongo? 1.Si , 0.No")
    if(deteccion_hongo=="1"):
        imagen_hongo_aislado=separacion_fondo(video,200)
        coordenadas_puntos_hongo=detectar_coordenadas_puntos_en_hongo(imagen_hongo_aislado)
        coordenadas_x=coordenadas_puntos_hongo[:,0]
        coordenadas_y=coordenadas_puntos_hongo[:,1]
        filas=frame.shape[0]
        columnas=frame.shape[1]
        
        #GRAFICO DE PUNTOS DE HONGO
        # plt.scatter(coordenadas_x,filas-coordenadas_y,s=2)
        # plt.xlim(0,columnas)
        # plt.ylim(0,filas)
        # plt.show()
        
    p=input("Hola")
    #Variables para ajustar que frames estudiar
    salto_frame = 1
    frame_contador = 1    
    while(True):
        is_true,frame=video.read()
        if is_true :
            
            #Mascara para enfocarse en el label elegido
            frame=aplicar_mascara_frame(frame, coordenadas_mascara)
            
            if (frame_contador%salto_frame==0):
                
                #Deteccion de hormigas
                coordenadas_hormigas=detectar_coordenadas_hormigas(frame)
                if len(coordenadas_hormigas)!=0:
                    
                    #Individualizacion de cada punto para detectar si es del label elegido (hormiga) o ruido
                    labels_hormigas=individualizar_hormigas(coordenadas_hormigas,8,20,average=True)
                    coordenadas_x=coordenadas_hormigas[:,0]
                    coordenadas_y=coordenadas_hormigas[:,1]
                    
                    #Reconocimiento de punto central
                    unicos_labels=np.unique(labels_hormigas)
                    #Si solo se reconoce un label (Probablemente la misma hormiga) calcula el punto central
                    if len(unicos_labels)==1:
                        label_punto_central=unicos_labels[0]
                        punto_central=conseguir_punto_central_frame(labels_hormigas,coordenadas_x,coordenadas_y,label_punto_central)
                        coordenadas_x_recorrido.append(punto_central[0])
                        coordenadas_y_recorrido.append(punto_central[1])
                        
                        
                    #Si se reconocen dos labels y el primero es ruido calcula el punto central del otro label (probablemente la hormiga)
                    elif (len(unicos_labels)==2 and unicos_labels[0]==-1):
                        label_punto_central=unicos_labels[1]
                        punto_central=conseguir_punto_central_frame(labels_hormigas,coordenadas_x,coordenadas_y,label_punto_central)
                        coordenadas_x_recorrido.append(punto_central[0])
                        coordenadas_y_recorrido.append(punto_central[1])
                        
                    #Si reconoce mas de un label , identifica el punto central de cada uno y escoge el mas cercano al punto central del pasado frame
                    else:
                        puntos_centrales_x=[]
                        puntos_centrales_y=[]
                        
                        #Si  hay ruido itera desde el segundo label , sino , itera desde el primer label
                        primer_index=1 if (unicos_labels[0]==-1) else 0
                        ultimo_index=len(unicos_labels) if (unicos_labels[0]==-1) else  len(unicos_labels)    
                        
                        #Identifica el punto central de cada label
                        for i in range(primer_index,ultimo_index):
                            punto_central_i=conseguir_punto_central_frame(labels_hormigas,coordenadas_x,coordenadas_y,unicos_labels[i])
                            puntos_centrales_x.append(punto_central_i[0])
                            puntos_centrales_y.append(punto_central_i[1])
                        
                        #Coordenadas de pasado punto central
                        punto_central_viejo_x=coordenadas_x_recorrido[-1]
                        punto_central_viejo_y=coordenadas_y_recorrido[-1]
                        
                        #Numpy array de las coordenadas de los puntos centrales de cada label
                        puntos_centrales_x=np.array(puntos_centrales_x)
                        puntos_centrales_y=np.array(puntos_centrales_y)
                        
                        #Distancia de cada punto central a el pasado punto central
                        distancias=((puntos_centrales_x - punto_central_viejo_x)**2 +  (puntos_centrales_y - punto_central_viejo_y)**2).tolist()
                        
                        #Establecimiento de nuevo punto central
                        index_punto_central_cercano_a_viejo=distancias.index(min(distancias))
                        coordenadas_x_recorrido.append(puntos_centrales_x[index_punto_central_cercano_a_viejo])
                        coordenadas_y_recorrido.append(puntos_centrales_y[index_punto_central_cercano_a_viejo])
                        punto_central=(coordenadas_x_recorrido[-1] , coordenadas_y_recorrido[-1] )
                        
                    
                    #Reubicar mascara a partir de la ubicacion del ultimo punto central
                    if len(coordenadas_x_recorrido)>=2:
                        x_viejo_nuevo_cpoints_distancia=coordenadas_x_recorrido[-1] - coordenadas_x_recorrido[-2]
                        y_viejo_nuevo_cpoints_distancia=coordenadas_y_recorrido[-1] - coordenadas_y_recorrido[-2]
                        coordenadas_mascara=[(coordenadas_mascara[0][0]+x_viejo_nuevo_cpoints_distancia,
                                              coordenadas_mascara[0][1]+y_viejo_nuevo_cpoints_distancia),
                                             (coordenadas_mascara[1][0]+x_viejo_nuevo_cpoints_distancia,
                                              coordenadas_mascara[1][1]+y_viejo_nuevo_cpoints_distancia)]
            
            frame_contador+=1
    
            filas=frame.shape[0]
            columnas=frame.shape[1]
            
            #GRAFICO  DE RECORRIDO DE LABEL (HORMIGA) SELECCIONADO
            
            # plt.scatter(coordenadas_x_recorrido,filas-np.array(coordenadas_y_recorrido),s=2)
            # plt.xlim(0,columnas)
            # plt.ylim(0,filas)
            # plt.show()
            
            
            #GRAFICO DE PUNTOS INDIVIDUALIZADOS CON PASADO PUNTO CENTRAL
            
            #colores = [(0.5, 0.5, 0.5) if label == -1 else (1, 0, 0) for label in labels_hormigas]
            # plt.scatter(coordenadas_x,filas-coordenadas_y,s=10,c=colores)
            # plt.xlim(0,columnas)
            # plt.ylim(0,filas)
            # if frame_contador>=3:
            #     plt.scatter(coordenadas_x_recorrido[-2],filas-coordenadas_y_recorrido[-2],s=10,c="green")
            # plt.show()
            
            #DESPLIGUE DE VIDEO CON CAPA Y PUNTO CENTRAL DE LABEL SELECCIONADO
            cv.circle(frame, punto_central, 2, (0,255,0),5)
            cv.imshow("Video",frame)
            if cv.waitKey(1) & 0xFF==ord('d'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()
    
    #GRAFICO  DE RECORRIDO DE LABEL (HORMIGA) SELECCIONADO
 
    video=cv.VideoCapture("ant_video.mp4")
    is_true,frame=video.read()
    filas=frame.shape[0]
    columnas=frame.shape[1]  
    plt.scatter(coordenadas_x_recorrido,filas-np.array(coordenadas_y_recorrido),s=2)
    plt.scatter(coordenadas_x_recorrido[0],filas - coordenadas_y_recorrido[0],s=2, c="green")
    plt.scatter(coordenadas_x_recorrido[-1],filas-coordenadas_y_recorrido[-1],s=2, c="red")
    plt.xlim(0,columnas)
    plt.ylim(0,filas)
    plt.show()
            
main()

