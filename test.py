from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Modèle léger, vous pouvez utiliser un modèle plus précis

# Définition des couleurs des capsules (HSV)
CAPSULE_COLORS = {
    "Eau Vive": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},  # Rouge
    "Cristaline": {"lower": np.array([90, 50, 50]), "upper": np.array([130, 255, 255])},  # Bleu
    "Fanta": {"lower": np.array([10, 100, 100]), "upper": np.array([25, 255, 255])}  # Orange
}

BOTTLE_CLASS_ID = 39  # Classe "bottle" dans le dataset COCO

def detecter_bouteilles(image_path, iou_threshold=0.6):
    """Détecte les bouteilles, dessine des contours et identifie leur marque selon la capsule."""
    
    # Charger l'image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Effectuer la détection avec YOLO
    results = model(image_rgb)

    # Stocker les détections
    boxes = []
    scores = []
    class_ids = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()
            class_id = int(class_id)

            if class_id == BOTTLE_CLASS_ID:  # Vérifier si c'est une bouteille
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)

    # Appliquer la suppression non maximale (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.3, nms_threshold=iou_threshold)

    # Stocker les résultats de détection des bouteilles
    bottles_count = defaultdict(int)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x1, y1, x2, y2 = box

            # Dessiner la boîte autour de la bouteille
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            # Extraire la région de la capsule (15% de la hauteur en haut)
            cap_height = int((y2 - y1) * 0.15)
            cap_roi = image[int(y1):int(y1 + cap_height), int(x1):int(x2)]

            # Identifier la capsule
            marque = identifier_capsule(cap_roi)

            if marque:
                bottles_count[marque] += 1

                # Dessiner un contour autour de la capsule
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y1 + cap_height)), (255, 0, 0), 2)

                # Ajouter le nom de la marque au-dessus de la capsule
                cv2.putText(image, f"{marque}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Sauvegarder l'image annotée
    output_path = "resultat_detection.jpg"
    cv2.imwrite(output_path, image)
    print(f"Résultat enregistré sous '{output_path}'.")

    # Afficher l'image
    img_pil = Image.open(output_path)
    img_pil.show()

    # Afficher les résultats en console
    for marque, count in bottles_count.items():
        print(f"{marque} : {count}")

    return bottles_count

def identifier_capsule(cap_roi):
    """Identifie la couleur dominante d'une capsule et retourne la marque correspondante."""

    if cap_roi is None or cap_roi.size == 0:
        return None

    # Convertir en HSV
    cap_hsv = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2HSV)

    # Détection de la couleur de la capsule
    for marque, color_range in CAPSULE_COLORS.items():
        mask = cv2.inRange(cap_hsv, color_range["lower"], color_range["upper"])
        if np.sum(mask) > 15000:  # Seuil pour éviter les faux positifs
            return marque

    return None  # Si aucune couleur ne correspond

# Exemple d'utilisation
if __name__ == "__main__":
    chemin_image = "salon-emploi-STAR.jpg"  # Remplacez par votre image
    resultats = detecter_bouteilles(chemin_image)
    print("Détails des bouteilles détectées :", resultats)
