import splitfolders
import os
import argparse

def split_data(input_folder, output_folder):
    # Safety Check: Does the input folder exist?
    if not os.path.exists(input_folder):
        print(f"❌ Error: The input folder '{input_folder}' does not exist.")
        return

    print(f"⏳ Splitting data from '{input_folder}' into Train (80%) and Val (20%)...")
    
    # This automatically shuffles and splits your images
    splitfolders.ratio(input_folder, output=output_folder,
                    seed=1337, ratio=(0.8, 0.2), group_prefix=None, move=False)
    
    print(f"✔ Done! Your data is ready at: {output_folder}")
    print("You can now train your model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into Train/Val sets for YOLO")
    
    parser.add_argument("--input", required=True, help="Path to the folder containing your class subfolders (dataset_final_piece)")
    parser.add_argument("--output", required=True, help="Path where the ready-to-train data will be saved")
    
    args = parser.parse_args()
    
    # Run the function with the provided arguments
    split_data(args.input, args.output)