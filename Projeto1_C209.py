
#Importando a biblioteca cv2, que detectará os objetos
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#Abrindo a imagem por meio da biblioteca cv2
#img = cv2.imread("royal_family_4.jpeg")
img = np.array(Image.open('royal_family_4.jpeg'))

# OpenCV abre a imagem em BGR,
# mas ela deve estar em RGB 
# para transformá-la em escala de cinza
# Utiliza-se funções próprias da cv2 para realizar a alteração
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Em vez de realizar as tranformações pela funções do cva
#realizou-se por meio do grayscale ensinado no laboratório
def grayscale(img_np):
    (l, c, p) = img_np.shape

    img_avg = np.zeros(shape=(l, c), dtype=np.uint8)
    for i in range(l):
        for j in range(c):
            r = float(img_np[i, j, 0])
            g = float(img_np[i, j, 1])
            b = float(img_np[i, j, 2])
        
            img_avg[i, j] = (r + g + b) / 3
     
    return img_avg

# Usa-se o minSize para que  
# pontos extras não interferiam 
# na análise da imagem
stop_data = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
img_gray = grayscale(img)
found = stop_data.detectMultiScale(img_gray, 
                                   minSize =(20,20))

# Caso não haja sinal,
# nada será feito
amount_found = len(found)

#Caso encontra-se algum sinal
if amount_found != 0:

    # Caso haja mais de um sinal, no caso rosto
    # marcará na imagem o retângulo verde
    for (x, y, width, height) in found:

        cv2.rectangle(img, (x, y), 
                      (x + height, y + width), 
                      (0, 255, 0), 5)

#Mostrando as imagens    
plt.figure(figsize=(16, 16))
plt.subplot(3, 1, 1)
plt.imshow(img)
plt.subplot(3, 1, 2)
plt.imshow(img_gray,cmap='gray')
