import cv2 as cv
import numpy as np
img=cv.imread("image.png") #Carga la imagen
img_RGB=cv.cvtColor(img,cv.COLOR_BGR2RGB) #Convierta la imagen de Bgt a Rgb para mayor comodidad
Dimension_img=img_RGB.shape #Dimensiones de imagen
img_2Dimension=img_RGB.reshape((-1,3)) #Aplana la matriz para ya no ubicar cada pixel por fila
valores_rojos=[]
for p in img_2Dimension: #recorre cada pixel 
    if (p[0]>(1.5*int(p[1])) and p[0]>(1.6*int(p[2]))): #Si  la componente r del pixel es mayor que la compontente g y b multiplicadas por 1.5 guarda el valor de r en valores_rojos
        p[0]=255 #Resalta el rojo
        p[1]=0
        p[2]=0      
print(valores_rojos) 
img_original = img_2Dimension.reshape(Dimension_img)  #Vuelve a la dimension ogiriginal
img_BGR= cv.cvtColor(img_original, cv.COLOR_RGB2BGR) #Pasa de rgb a bgr
cv.imshow("img",img_BGR) # Muestra la imagen
cv.waitKey(0)
