
# import numpy as np

# # Exemple : couleur orange Fanta en BGR
# orange_bgr = np.uint8([[[0, 140, 255]]])  # Bleu 0, Vert 140, Rouge 255

# # Convertir en HSV avec OpenCV
# orange_hsv = cv2.cvtColor(orange_bgr, cv2.COLOR_BGR2HSV)

# # Afficher les valeurs HSV
# print("HSV OpenCV:", orange_hsv[0][0])
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread("fanta1.jpeg")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Séparer les canaux HSV
H, S, V = cv2.split(image_hsv)

# Afficher les canaux individuellement
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(H, cmap="hsv")
ax[0].set_title("Teinte (H)")
ax[0].axis("off")

ax[1].imshow(S, cmap="gray")
ax[1].set_title("Saturation (S)")
ax[1].axis("off")

ax[2].imshow(V, cmap="gray")
ax[2].set_title("Valeur (V)")
ax[2].axis("off")

plt.show()

