import os
import cv2
import argparse # <--- 1. Import the library

def extract_empty_squares(input_folder, output_folder):
    # 2. We now build paths based on what the user gave us
    images_folder = os.path.join(input_folder, "images")
    labels_folder = os.path.join(input_folder, "labels")

    # Safety Check: Do these folders actually exist?
    if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
        print(f"❌ Error: Could not find 'images' or 'labels' inside: {input_folder}")
        return

    # Create the output folder if it doesn't exist
    save_folder = os.path.join(output_folder, "empty")
    os.makedirs(save_folder, exist_ok=True)
    
    # Get all images
    all_files = os.listdir(images_folder)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"📂 Found {len(image_files)} images in {images_folder}")
    
    count_saved = 0

    for filename in image_files:
        img_path = os.path.join(images_folder, filename)
        
        # Robust filename matching
        base_name = os.path.splitext(filename)[0] 
        label_filename = base_name + ".txt"
        label_path = os.path.join(labels_folder, label_filename)

        if not os.path.exists(label_path):
            print(f"⚠ Warning: No label file for {filename}. Skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        
        h_img, w_img, _ = img.shape
        square_w = w_img / 8.0 
        square_h = h_img / 8.0

        # Map Occupied Squares
        occupied_squares = set()
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                xc = float(parts[1])
                yc = float(parts[2])

                col = int(xc * 8)
                row = int(yc * 8)
                occupied_squares.add((row, col))

        # Extract ONLY Empty Squares
        for row in range(8):
            for col in range(8):
                if (row, col) not in occupied_squares:
                    x1 = int(col * square_w)
                    y1 = int(row * square_h)
                    x2 = int((col + 1) * square_w)
                    y2 = int((row + 1) * square_h)

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    save_name = f"{base_name}_r{row}c{col}.jpg"
                    cv2.imwrite(os.path.join(save_folder, save_name), crop)
                    count_saved += 1

    print(f"✔ Success! Extracted {count_saved} empty squares to: {save_folder}")

if __name__ == "__main__":
    # 3. This is the "Menu" logic
    parser = argparse.ArgumentParser(description="Tool to extract empty chess squares")
    
    # We define two "slots" for information
    parser.add_argument("--input", required=True, help="Path to the folder containing images/ and labels/")
    parser.add_argument("--output", required=True, help="Path where the empty squares will be saved")
    
    # Read the arguments from the terminal
    args = parser.parse_args()
    
    # Pass the information to the Chef (your function)
    extract_empty_squares(args.input, args.output)