import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# === Fonctions utilitaires ===
def vider_dossier(dossier):
    if os.path.exists(dossier):
        for f in os.listdir(dossier):
            chemin = os.path.join(dossier, f)
            if os.path.isfile(chemin):
                os.remove(chemin)
    else:
        os.makedirs(dossier)

# === 0. Définition des chemins ===
video_path = '/Users/flobaillien/DocumentsPC/HEPL/Machine-learning/Projet_Machine_Learning_Air_Drawing/convert/videos/Lettres/L.mp4'
extracted_dir = '/Users/flobaillien/DocumentsPC/HEPL/Machine-learning/Projet_Machine_Learning_Air_Drawing/convert/images_extraites'
finger_dir = '/Users/flobaillien/DocumentsPC/HEPL/Machine-learning/Projet_Machine_Learning_Air_Drawing/convert/finger_find'
frame_interval = 5

# === 1. Nettoyage des dossiers ===
vider_dossier(extracted_dir)
vider_dossier(finger_dir)

# === 2. Extraction des frames ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erreur : impossible d'ouvrir la vidéo '{video_path}'")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo chargée : {total_frames} frames à {fps:.2f} fps")

frame_count = 0
saved_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(extracted_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"{saved_count} images extraites dans le dossier '{extracted_dir}'")

# === 3. Détection du bout de l'index ===
detector = HandDetector(staticMode=True, maxHands=1, detectionCon=0.7)

for filename in os.listdir(extracted_dir):
    if not filename.endswith(('.jpg', '.png')):
        continue

    image_path = os.path.join(extracted_dir, filename)
    image = cv2.imread(image_path)
    hands, img = detector.findHands(image)

    if hands:
        hand = hands[0]
        lm_list = hand['lmList']
        if len(lm_list) >= 9:
            x, y = lm_list[8][0], lm_list[8][1]
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(finger_dir, filename), img)

# === 4. Génération de l'image composite ===
sample_img = cv2.imread(os.path.join(finger_dir, os.listdir(finger_dir)[0]))
height, width, _ = sample_img.shape
result = np.zeros((height, width, 3), dtype=np.uint8)

for filename in os.listdir(finger_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(finger_dir, filename)
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) > 0:
                cv2.drawContours(result, [biggest], -1, (0, 0, 255), -1)

# === 5. Rotation de 90° vers la droite ===
rotated = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

# === 6. Effet miroir (symétrie horizontale) ===
mirrored = cv2.flip(rotated, 1)

# === 7. Sauvegarde de l'image finale ===
cv2.imwrite("image_resultat.png", mirrored)
print("Image finale enregistrée sous 'image_resultat.png' (rotation + effet miroir)")
