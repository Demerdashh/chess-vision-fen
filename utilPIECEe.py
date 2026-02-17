import cv2
import os
import glob
import argparse

# ---------------- CONFIG ----------------
# Roboflow classes names mapping (Constant)
# These are specific to the dataset model, so we keep them hardcoded here.
CLASS_NAMES = [
    'WB',  # 0 -> B (white bishop)
    'WK',  # 1 -> K (white king)
    'WN',  # 2 -> N (white knight)
    'WP',  # 3 -> P (white pawn)
    'WQ',  # 4 -> Q (white queen)
    'WR',  # 5 -> R (white rook)
    'BB',  # 6 -> b (black bishop)
    'board', # 7 -> board (we usually ignore this for piece classification, but kept here for index alignment)
    'BK',  # 8 -> k (black king)
    'BN',  # 9 -> n (black knight)
    'BP',  # 10 -> p (black pawn)
    'BQ',  # 11 -> q (black queen)
    'BR'   # 12 -> r (black rook)
]
# ----------------------------------------

def harvest_pieces(dataset_path, output_root):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    # Safety Check
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"❌ Error: Could not find 'images' or 'labels' folders inside: {dataset_path}")
        return

    # Create output folders for each class
    for name in CLASS_NAMES:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    # Find images
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_path, ext)))
    
    print(f"🚜 Harvesting pieces from {len(image_files)} images in {dataset_path}...")

    count_saved = 0

    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None: continue

        h_img, w_img, _ = img.shape

        basename = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(labels_path, basename + ".txt")

        if not os.path.exists(label_file): continue

        with open(label_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            cls_id = int(parts[0])
            
            # Safety check: Ignore classes outside our list
            if cls_id >= len(CLASS_NAMES): continue

            # YOLO (Normalized) -> Pixel Coords
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            x1 = int((x_c - w/2) * w_img)
            y1 = int((y_c - h/2) * h_img)
            x2 = int((x_c + w/2) * w_img)
            y2 = int((y_c + h/2) * h_img)

            # Clamp coordinates to image boundaries (prevents crashes)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            # Crop & Save
            crop = img[y1:y2, x1:x2]
            
            if crop.size > 0:
                save_name = os.path.join(output_root, CLASS_NAMES[cls_id], f"{basename}_{i}.jpg")
                cv2.imwrite(save_name, crop)
                count_saved += 1

    print(f"✔ Harvest complete! Extracted {count_saved} pieces to '{output_root}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual chess pieces from a YOLO dataset")
    
    parser.add_argument("--input", required=True, help="Path to the dataset train folder (containing images/ and labels/)")
    parser.add_argument("--output", required=True, help="Folder where cropped pieces will be saved")
    
    args = parser.parse_args()
    
    harvest_pieces(args.input, args.output)