import cv2
import numpy as np
from moondream import ColorAnalyzer
import argparse

class BouteilleDetector:
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.adaptive_threshold = True
        
    def detecter_bouteille(self, image):
        # Conversion en HSV pour l'analyse des couleurs
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Détection des contours de la bouteille
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Utilisation de la méthode d'Otsu pour le seuillage automatique
        _, thresh = cv2.threshold(blurred, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Trouver les contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrer les petits contours
            if area > 1000:
                # Extraire la région d'intérêt (ROI) autour du contour
                roi = image[y:y+h, x:x+w]
                hsv_roi = hsv[y:y+h, x:x+w]
                
                # Analyse adaptative des couleurs
                if self.adaptive_threshold:
                    self._analyse_couleur_adaptive(hsv_roi)
                else:
                    self._analyse_couleur_classique(hsv_roi)
                    
                # Déterminer le type de bouteille
                return self._determiner_type_bouteille()
                    
        return "Type non identifié"
    
    def _analyse_couleur_adaptive(self, hsv_roi):
        # Calculer les histogrammes pour chaque canal HSV
        h_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
        
        # Trouver les pics dans l'histogramme de teinte (H)
        h_peaks = self._trouver_pics(h_hist)
        
        # Déterminer la couleur dominante
        self.couleur_dominante = self._analyser_pics(h_peaks)
    
    def _analyse_couleur_classique(self, hsv_roi):
        # Utilisation de la méthode d'Otsu pour chaque canal
        h_thresh = cv2.threshold(hsv_roi[:,:,0], 0, 179, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        s_thresh = cv2.threshold(hsv_roi[:,:,1], 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        v_thresh = cv2.threshold(hsv_roi[:,:,2], 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Analyse des seuils obtenus
        self._analyser_seuils(h_thresh, s_thresh, v_thresh)
    
    def _trouver_pics(self, hist):
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
    
    def _analyser_pics(self, peaks):
        if not peaks:
            return "Inconnu"
        
        # Déterminer la couleur basée sur les pics
        principal_peak = peaks[0]
        if 0 <= principal_peak <= 10:
            return "Rouge"
        elif 10 < principal_peak <= 20:
            return "Orange"
        # Ajouter d'autres plages de couleurs si nécessaire
    
    def _analyser_seuils(self, h_thresh, s_thresh, v_thresh):
        # Analyse des valeurs moyennes des seuils
        h_moyen = np.mean(h_thresh)
        s_moyen = np.mean(s_thresh)
        v_moyen = np.mean(v_thresh)
        
        # Classification basée sur les valeurs moyennes
        if h_moyen < 10 and s_moyen > 100:
            self.couleur_dominante = "Rouge"
        elif h_moyen < 20 and s_moyen > 100:
            self.couleur_dominante = "Orange"
    
    def _determiner_type_bouteille(self):
        if self.couleur_dominante == "Rouge":
            return "Eau Vive" if self.color_analyzer.analyze_transparency() else "Eau Cristalline"
        elif self.couleur_dominante == "Orange":
            return "Fanta"
        return "Type non identifié"

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Détecteur de bouteilles')
    parser.add_argument('--image', type=str, required=True,
                       help='images.png')
    parser.add_argument('--adaptive', action='store_true',
                       help='Activer le seuillage adaptatif')
    args = parser.parse_args()
    
    # Initialisation du détecteur
    detector = BouteilleDetector()
    detector.adaptive_threshold = args.adaptive
    
    # Charger l'image
    image = cv2.imread(args.image)
    if image is None:
        print("Erreur: Impossible de charger l'image")
        return
    
    # Détecter la bouteille
    resultat = detector.detecter_bouteille(image)
    
    # Afficher le résultat
    cv2.putText(image, resultat, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)
    cv2.imshow('Detection Bouteille', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()