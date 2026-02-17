import cv2
from ultralytics import YOLO
import os

model = YOLO("runs/detect/board_train/weights/best.pt")


def crop_board(image_path, model):
    """
    Detects the chessboard and returns the cropped board image.
    Model is already loaded globally and passed here.
    """

    results = model(image_path)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("❌ No board detected.")
        return None

    # Take the first detected board
    x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy().astype(int)

    img = cv2.imread(image_path)
    cropped = img[y1:y2, x1:x2]

    return cropped


def normalize_board(board_img, size=480):
    """Resize board to a square of fixed size."""
    return cv2.resize(board_img, (size, size))


def split_into_squares(board_img, output_folder="squares", size=480):
    """Split the normalized board into 64 equal squares."""

    os.makedirs(output_folder, exist_ok=True)
    square_size = size // 8

    for row in range(8):
        for col in range(8):
            y1 = row * square_size
            y2 = (row + 1) * square_size
            x1 = col * square_size
            x2 = (col + 1) * square_size

            square = board_img[y1:y2, x1:x2]
            cv2.imwrite(f"{output_folder}/{row}_{col}.png", square)

    print("✔ 64 squares saved!")


# --------------------------------
# MAIN PIPELINE
# --------------------------------
if __name__ == "__main__":
    image_path = "phpbUkXp2.jpg"

    board = crop_board(image_path, model)

    if board is not None:
        cv2.imwrite("board_cropped.png", board)
        print("✔ Board cropped → board_cropped.png")

        board_norm = normalize_board(board)
        cv2.imwrite("board_normalized.png", board_norm)
        print("✔ Board normalized → 480×480")

        split_into_squares(board_norm, output_folder="squares")
