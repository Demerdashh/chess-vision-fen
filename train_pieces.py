from ultralytics import YOLO
import argparse

def train_model(data_path, epochs, device):
    print(f"🚀 Starting training on dataset: {data_path}")
    print(f"⚙ Settings: Epochs={epochs}, Device={device}")

    # load classification model backbone
    model = YOLO("yolov8n-cls.pt") 

    model.train(
        data=data_path,
        epochs=epochs,
        batch=64,
        imgsz=64,             # square input size
        name="piece_classifier_v1",
        device=device,        # "cpu" or 0 for GPU
        patience=5            # early stopping
    )
    
    print("✔ Training Complete. Check the 'runs' folder for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Piece Classifier")
    
    # Required argument: Where is the data?
    parser.add_argument("--data", required=True, help="Path to the dataset_ready_for_yolo folder")
    
    # Optional arguments (with defaults)
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--device", default="cpu", help="Device to use: 'cpu' or '0' (for GPU)")
    
    args = parser.parse_args()
    
    train_model(args.data, args.epochs, args.device)