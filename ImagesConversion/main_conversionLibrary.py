import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def save_step_logs(image, name, output_dir="debug_steps"):
    os.makedirs(output_dir, exist_ok=True)
    if image is None or image.size == 0:
        print(f"[WARNING] Cannot save '{name}': image is empty.")
        return
    cv2.imwrite(os.path.join(output_dir, f"{name}.png"), image)

# Step 1: Extract red from image
def extract_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

# Step 2: Basic cleaning (open/close)
def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Step 3: Keep relevant parts
def filter_components(mask, min_area=50):
    labeled = label(mask)
    cleaned = np.zeros_like(mask)
    for region in regionprops(labeled):
        if region.area >= min_area:
            for y, x in region.coords:
                cleaned[y, x] = 255
    return cleaned

# Step 4: Skeletonize
def get_skeleton(mask):
    return (skeletonize(mask > 0) * 255).astype(np.uint8)

# Step 5 : Bold the ligne
def thicken_mask(mask, size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)

# Step 6: Resize and center
def center_and_resize(mask, output_size=28, margin=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((output_size, output_size), dtype=np.uint8) * 255
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = mask[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (output_size - 2 * margin, output_size - 2 * margin))
    canvas = np.ones((output_size, output_size), dtype=np.uint8) * 255
    cx = (output_size - resized.shape[1]) // 2
    cy = (output_size - resized.shape[0]) // 2
    canvas[cy:cy + resized.shape[0], cx:cx + resized.shape[1]] = 255 - resized

    return canvas

# Main function
def process_image(path, output_dir="debug_steps"):
    img = cv2.imread(path)
    save_step_logs(img, "00_original", output_dir)

    mask = extract_red_mask(img)
    save_step_logs(mask, "01_red_mask", output_dir)

    cleaned = clean_mask(mask)
    save_step_logs(cleaned, "02_cleaned", output_dir)

    filtered = filter_components(cleaned)
    save_step_logs(filtered, "03_filtered", output_dir)

    skeleton = get_skeleton(filtered)
    save_step_logs(skeleton, "04_skeleton", output_dir)

    thickened = thicken_mask(skeleton, size=50)
    save_step_logs(thickened, "05_thickened", output_dir)

    final = center_and_resize(thickened)
    save_step_logs(final, "06_final", output_dir)

    print("[INFO] Simplified processing complete.")
    cv2.imwrite(os.path.join("../Resultats/Conversion", "result.png"), final)

    return final

# Run on your image
process_image("../image_resultat.png")