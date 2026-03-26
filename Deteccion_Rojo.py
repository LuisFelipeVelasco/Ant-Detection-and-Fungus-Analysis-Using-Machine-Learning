import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

img=cv.imread("image.png")
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV) #Hsv contiene color,saturacion y brillo por lo que es un espacio de tres capas
pixeles = hsv[:, :, 0].reshape(-1, 1) # hsv[:,:,0]Toma toda la imagen y selecciona solo la capa de color (#0) y reshape(-1, 1) convierte la matriz de píxeles en una matriz de una sola columna, donde cada fila representa el valor de color de un píxel.  
kmeans = KMeans(n_clusters=10) #Crea una instancia de KMeans con 10 clusters, lo que significa que el algoritmo intentará agrupar los píxeles en 10 grupos diferentes según sus valores de color.
kmeans.fit(pixeles) #Ajusta el modelo KMeans a los datos de píxeles, es decir asigna cada píxel a uno de los clusters y calcular los centroides de cada cluster.
labels = kmeans.labels_ #Obtiene los clusters a los que pertenece cada pixel
centroides = kmeans.cluster_centers_ #Obtiene los centroides 
#En la escala hsv el rojo es el primer y ultmo valor , por lo que el cluster mas pequeños y grande probablemente corresponda al rojo
primer_cluster_rojo = np.argmin(centroides[:, 0])   
segundo_cluster_rojo = np.argmax(centroides[:, 0])
#mascaras o vectores booleanos que indican si un pixel pertenece a uno de los clusteres rojos o no
brillo_pixeles=hsv[:,:,2].reshape(-1,1) #Obtenemos la capa de brillo de la imagen y la convertimos en una matriz de una sola columna, donde cada fila representa el valor de brillo de un píxel.
mask_1_brillo=(brillo_pixeles > 100).flatten() #Creamos una mascara que indica si el brillo de un pixel es mayor a 100, esto se hace para evitar que los pixeles muy oscuros sean considerados como rojos
mask_1_rojo = (labels == primer_cluster_rojo)  
mask_2_rojo = (labels == segundo_cluster_rojo)
#En caso de que pertenezca a alguno de los cluteres rojos y su brillo sea mayor a 100 se considera como un pixel rojo
mask_color = (mask_1_rojo | mask_2_rojo) & (mask_1_brillo)
#Mask deja de ser una lista y se convierte en una matriz de las dimiensiones de la imagen original 
mask = mask_color.reshape(img.shape[:2])
#Result es una copia de la imagen
resultado = img.copy()
#De resultado los pixeles que tienen un valor verdadero en la mascara que es lo rojo , se resalta con el color rojo puro
resultado[mask] = [0, 0, 255]

cv.imshow("Deteccion Rojo", resultado)
cv.waitKey(0) 