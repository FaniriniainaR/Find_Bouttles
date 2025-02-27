from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO("yolov8n.pt")  # Modèle léger, changez par 'yolov8m.pt' pour plus de précision

# Définir les classes qui nous intéressent (bouteilles appartiennent aux "bottles" dans COCO dataset)
BOTTLE_CLASS_ID = 39  # Identifiant de la classe "bottle" dans COCO dataset

def detecter_bouteilles(image_path, iou_threshold=0.6):
    """Détecte les bouteilles dans une image en utilisant YOLOv8 et compte les bouteilles sans doublons."""

    # Charger l'image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB pour l'affichage

    # Effectuer la détection
    results = model(image_rgb)

    # Extraire les boîtes de détection et les scores
    boxes = []
    scores = []
    class_ids = []

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.tolist()
            if int(class_id) == BOTTLE_CLASS_ID:  # Vérifier si c'est une bouteille
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(score)
                class_ids.append(int(class_id))

    # Appliquer la Non-Maximum Suppression (NMS)
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.1, nms_threshold=iou_threshold)

    count_bottles = 0
    if len(indices) > 0:
        # Compter les bouteilles après NMS
        for i in indices.flatten():
            count_bottles += 1
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, f"Bouteille {scores[i]:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Sauvegarder et afficher l'image avec les détections
    output_path = "resultat_detection.jpg"
    cv2.imwrite(output_path, image)
    print(f"Résultat enregistré sous '{output_path}'.")

    # Afficher l'image
    img_pil = Image.open(output_path)
    img_pil.show()

    # Afficher le nombre total de bouteilles détectées
    print(f"Nombre total de bouteilles détectées : {count_bottles}")
    return count_bottles

# Exemple d'utilisation
if __name__ == "__main__":
    chemin_image = "images.png"  # Remplacez par votre image
    count = detecter_bouteilles(chemin_image)
    print(f"Le nombre total de bouteilles détectées dans l'image est : {count}")
