import cv2
import os

video_path = '/Users/flobaillien/DocumentsPC/HEPL/Machine-learning/Projet_Machine_Learning_Air_Drawing/convert/videos/test.mp4'
output_dir = '/Users/flobaillien/DocumentsPC/HEPL/Machine-learning/Projet_Machine_Learning_Air_Drawing/convert/images_extraites'  # Crée ce dossier à côté du script
frame_interval = 30  # Une image toutes les 30 frames

os.makedirs(output_dir, exist_ok=True)

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
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"{saved_count} images extraites dans le dossier '{output_dir}'")
