from ultralytics import YOLO
import os

# Ensure output directory exists
output_folder = r"C:\Users\User\Desktop\do\Yolo\output"  # Use raw string for file path
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
model = YOLO('yolov8m.pt')

# Input file path
input_file = r"C:\Users\User\Desktop\do\Yolo\images\toyota.jpg"

# Check if the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

# Perform object detection
results = model.predict(
    source=input_file,  # Use raw string for file path
    save=True,
    conf=0.25,
    save_dir=output_folder  # Specify where to save results
)

# Print results
print("Detection complete. Results saved in:", output_folder)
