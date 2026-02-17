# main.py
"""
Chess FEN Generator - 2-Stage YOLOv8 Pipeline
Converts a screenshot (or photo) of a chessboard into a FEN string.

Usage:
    python main.py --input path/to/image.png
    python main.py -i screenshot.png --visualize

Notes:
 - Expects detection weights at: models/detect/board_train/weights/best.pt
 - Expects classification weights at: models/classify/piece_classifier_v1/weights/best.pt
 - Adjust BOARD_SIZE / CLASSIFIER_INPUT_SIZE to match your training choices.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# CONFIG
# -------------------------
DETECTOR_WEIGHTS = "models/detect/board_train/weights/best.pt"
CLASSIFIER_WEIGHTS = "models/classify/piece_classifier_v1/weights/best.pt"

# Normalized board size (must be divisible by 8)
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

# Classifier input size used during training (change if you trained with different size)
CLASSIFIER_INPUT_SIZE = 64

# Class name -> FEN char mapping (must match classifier training names)
CLASS_TO_FEN = {
    "WK": "K", "WQ": "Q", "WR": "R", "WB": "B", "WN": "N", "WP": "P",
    "BK": "k", "BQ": "q", "BR": "r", "BB": "b", "BN": "n", "BP": "p",
    "empty": "1"
}

# -------------------------
# UTILITIES
# -------------------------
def load_models(detector_path: str, classifier_path: str) -> Tuple[YOLO, YOLO]:
    """Load YOLO detector and classifier from given paths."""
    dpath = Path(detector_path)
    cpath = Path(classifier_path)
    if not dpath.exists():
        raise FileNotFoundError(f"Detector weights not found: {dpath}")
    if not cpath.exists():
        raise FileNotFoundError(f"Classifier weights not found: {cpath}")

    print(f"Loading detector from: {dpath}")
    detector = YOLO(str(dpath))
    print(f"Loading classifier from: {cpath}")
    classifier = YOLO(str(cpath))

    return detector, classifier

def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int,int,int,int]:
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    return x1, y1, x2, y2

# -------------------------
# STAGE 1: BOARD DETECTION
# -------------------------
def detect_board(image: np.ndarray, detector: YOLO) -> Tuple[Optional[np.ndarray], Optional[float], Optional[Tuple[int,int,int,int]]]:
    """
    Detect the chessboard in the input image and return the board crop,
    the confidence of the chosen detection, and the bounding box (x1,y1,x2,y2).
    If no board found returns (None, None, None).
    """

    results = detector(image, verbose=False)
    if len(results) == 0:
        return None, None, None
    res = results[0]
    boxes = res.boxes
    if len(boxes) == 0:
        return None, None, None

    # Choose the box with highest confidence
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    best_conf = float(confs[best_idx])

    xyxy_all = boxes.xyxy.cpu().numpy()  # shape (N,4)
    x1, y1, x2, y2 = xyxy_all[best_idx].astype(int)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return None, None, None

    crop = image[y1:y2, x1:x2].copy()

    return crop, best_conf, (x1, y1, x2, y2)

# -------------------------
# PREPROCESS: NORMALIZE + SPLIT
# -------------------------
def normalize_board(board_img: np.ndarray, size: int = BOARD_SIZE) -> np.ndarray:
    """Resize board crop to (size x size)."""
    return cv2.resize(board_img, (size, size), interpolation=cv2.INTER_LINEAR)

def split_board_into_squares(board_img: np.ndarray) -> List[np.ndarray]:
    """
    Split normalized board image into 64 square images.
    Order: top-left -> top-right, next row, ... (row-major).
    This corresponds to FEN order: a8,h8, a7,h7, ... a1,h1 when interpreted appropriately.
    """
    squares = []
    s = SQUARE_SIZE
    for r in range(8):
        for c in range(8):
            y1 = r * s
            y2 = (r + 1) * s
            x1 = c * s
            x2 = (c + 1) * s
            sq = board_img[y1:y2, x1:x2].copy()
            squares.append(sq)

    return squares

# -------------------------
# STAGE 2: CLASSIFICATION
# -------------------------
def classify_squares(squares: List[np.ndarray], classifier: YOLO, input_size: int = CLASSIFIER_INPUT_SIZE) -> List[str]:
    """
    Classify list of 64 squares using the classification YOLO model.
    Returns list of class names (strings) in same order as input squares.
    """
    if len(squares) == 0:
        return []

    # Resize squares to classifier input size
    prepared = [cv2.resize(sq, (input_size, input_size)) for sq in squares]

    # Pass the LIST directly - do NOT convert to np.array()
    # YOLO accepts a list of numpy arrays for batch inference
    results = classifier(prepared, verbose=False)

    class_names = []
    for r in results:
        if hasattr(r, "probs") and r.probs is not None:
            # Use the .top1 attribute to get the predicted class index
            idx = int(r.probs.top1)
        else:
            idx = 0
        
        # Get class name from the names dictionary
        names_map = r.names if hasattr(r, "names") else classifier.names
        class_name = names_map.get(idx, "empty")
        class_names.append(class_name)
    
    return class_names

# -------------------------
# FEN BUILDING
# -------------------------
def compress_fen_row(row_chars: List[str]) -> str:
    """Compress 8 characters into a FEN row string (compress empty '1's)."""
    out = []
    empty_count = 0
    for ch in row_chars:
        if ch == "1":
            empty_count += 1
        else:
            if empty_count > 0:
                out.append(str(empty_count))
                empty_count = 0
            out.append(ch)
    if empty_count > 0:
        out.append(str(empty_count))
    return "".join(out)

def build_fen_from_predicted_chars(pred_chars: List[str]) -> str:
    """
    pred_chars: list of 64 FEN-char-equivalents like 'P','k','1' ordered row-major top->bottom.
    Returns FEN position string (piece placement only).
    """
    if len(pred_chars) != 64:
        raise ValueError("Expected 64 characters to build FEN, got: {}".format(len(pred_chars)))
    rows = []
    for r in range(8):
        start = r * 8
        row = pred_chars[start:start+8]
        rows.append(compress_fen_row(row))
    position = "/".join(rows)
    return position

# -------------------------
# ORIENTATION HELPER
# -------------------------
def detect_and_fix_orientation(class_grid: List[List[str]]) -> Tuple[List[List[str]], bool]:
    """
    Heuristic: if white pieces are on top and black on bottom, flip vertically so white is on bottom.
    Input grid: list of 8 rows top->bottom, each row list of 8 class names like 'WP','empty','BK',...
    Returns possibly flipped grid and boolean indicating whether flip occurred.
    """
    white_rows = []
    black_rows = []
    
    for r in range(8):
        for c in range(8):
            cls = class_grid[r][c]
            if cls.startswith("W"):
                white_rows.append(r)
            elif cls.startswith("B"):
                black_rows.append(r)
    if len(white_rows) == 0 or len(black_rows) == 0:
        return class_grid, False
    mean_white = sum(white_rows) / len(white_rows)
    mean_black = sum(black_rows) / len(black_rows)
    # if white average row index < black average -> white nearer top -> flip
    if mean_white < mean_black:
        
        print("black")

        flipped_rows = class_grid[::-1]
        fully_rotated = [row[::-1] for row in flipped_rows]

        return fully_rotated, True
    else :
        print("white")
    return class_grid, False

# -------------------------
# VISUALIZATION (optional)
# -------------------------
def display_debug(original_img: np.ndarray, bbox: Tuple[int,int,int,int], board_img: np.ndarray, squares: List[np.ndarray], fen_chars: List[str], fen_str: str):
    """Show windows for debugging: detected bbox, normalized board with overlay, ASCII print."""
    vis = original_img.copy()
    if bbox:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 3)
    cv2.imshow("Original with detection", cv2.resize(vis, (1000, int(1000 * vis.shape[0] / vis.shape[1]))))
    # Board with grid and labels
    board_vis = normalize_board(board_img, size=BOARD_SIZE)
    for i in range(1,8):
        cv2.line(board_vis, (i*SQUARE_SIZE,0), (i*SQUARE_SIZE, BOARD_SIZE), (0,255,0), 1)
        cv2.line(board_vis, (0,i*SQUARE_SIZE), (BOARD_SIZE, i*SQUARE_SIZE), (0,255,0), 1)
    # overlay predicted FEN chars (non-empty) in center of each square
    for idx, ch in enumerate(fen_chars):
        if ch != "1":
            r = idx // 8
            c = idx % 8
            x = c*SQUARE_SIZE + SQUARE_SIZE//2 - 10
            y = r*SQUARE_SIZE + SQUARE_SIZE//2 + 10
            cv2.putText(board_vis, ch, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow("Board predictions", cv2.resize(board_vis, (600,600)))
    # print matrix to console
    print("\nPredicted board (top->bottom rows):")
    for r in range(8):
        row = fen_chars[r*8:(r+1)*8]
        print(" ".join([c if c!="1" else "." for c in row]))
    print("\nFEN:", fen_str)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------
# MAIN PIPELINE
# -------------------------
def process_image_to_fen(image_path: str, detector: YOLO, classifier: YOLO, visualize: bool = False) -> str:
    # Load image
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError("Failed to read image with OpenCV.")

    # 1) detect board
    board_crop, conf, bbox = detect_board(img, detector)
    if board_crop is None:
        raise ValueError("No board detected in the image.")
    # 2) normalize
    board_norm = normalize_board(board_crop, size=BOARD_SIZE)
    # 3) split into squares
    squares = split_board_into_squares(board_norm)
    # 4) classify squares -> class names like 'WP','empty','BK',...
    class_names = classify_squares(squares, classifier, input_size=CLASSIFIER_INPUT_SIZE)
    # 5) class_names -> FEN characters
    # Convert to FEN-character list (64 elements) using CLASS_TO_FEN mapping
    fen_chars = []
    for name in class_names:
        fen_char = CLASS_TO_FEN.get(name, "1")  # default to empty on unknown
        fen_chars.append(fen_char)
    # 6) Optional orientation fix (operate on class names grid)
    # Build 8x8 grid of original class names first (top->bottom)
    grid = []
    for r in range(8):
        grid.append(class_names[r*8:(r+1)*8])
    grid_fixed, flipped = detect_and_fix_orientation(grid)
    # Convert fixed grid back to fen_chars order (top->bottom)
    fixed_fen_chars = []
    for r in range(8):
        for c in range(8):
            cls_name = grid_fixed[r][c]
            fixed_fen_chars.append(CLASS_TO_FEN.get(cls_name, "1"))
    # 7) Build FEN position string
    fen_position = build_fen_from_predicted_chars(fixed_fen_chars)
    # Return final FEN (piece placement). If you want full FEN add " w - - 0 1" etc.
    full_fen = f"{fen_position} w - - 0 1"
    # visualize if requested
    if visualize:
        display_debug(img, bbox, board_norm, squares, fixed_fen_chars, fen_position)
    return full_fen

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert chessboard image to FEN using 2-stage YOLOv8 pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--detector", default=DETECTOR_WEIGHTS, help="Path to board detector weights")
    parser.add_argument("--classifier", default=CLASSIFIER_WEIGHTS, help="Path to piece classifier weights")
    parser.add_argument("--visualize", action="store_true", help="Show debug visualization windows")
    args = parser.parse_args()

    try:
        detector, classifier = load_models(args.detector, args.classifier)
    except FileNotFoundError as e:
        print("Error loading models:", e, file=sys.stderr)
        sys.exit(1)

    try:
        fen = process_image_to_fen(args.input, detector, classifier, visualize=args.visualize)
    except Exception as e:
        print("Processing error:", e, file=sys.stderr)
        sys.exit(1)

    print("\n=== RESULT ===")
    print(fen)
    # also print ASCII board for quick human reading (optional)
    # simple ASCII row print
    rows = fen.split()[0].split("/")
    print("\nBoard (FEN rows top->bottom):")
    for r, row in enumerate(rows):
        print(f"{8-r}  {row}")
    print("   a b c d e f g h")
    sys.exit(0)

if __name__ == "__main__":
    main()
