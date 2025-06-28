# Food and Calorie Estimation

A computer vision application that detects food items from images or video streams and estimates their calorie content in real-time.

![Food Detection Example](https://via.placeholder.com/800x400?text=Food+Detection+Example)

## Overview

This project uses deep learning (YOLO model) to identify Indian food items from images or video streams, and provides nutritional information including calorie content. It features:

- Real-time food detection from webcam feed
- Calorie estimation based on detected foods
- Nutritional information display (calories, protein, fat, etc.)
- Calorie intake tracking and history
- Interactive data visualization of consumption patterns

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended but not required)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/food-and-calorie-estimation.git
   cd food-and-calorie-estimation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   ```bash
   # The model file (model.pt) should already be in the repository
   # If not, download it from the releases page
   ```

## Usage

### Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

Navigate to the local URL displayed in your terminal (typically http://localhost:8501).

### Features

- **Real-time Detection**: Use your webcam to detect food items in real-time
- **Image Upload**: Upload images for food detection
- **Calorie Tracking**: View your daily and historical calorie intake
- **Nutritional Information**: Get detailed nutritional breakdown of detected foods

## Model Training

### Dataset

The model was trained on a custom dataset of Indian foods with 20 classes including:
- Aloo_matar
- Besan_cheela
- Biryani
- Chapathi
- Chole_bature
- And more...

### Training Process

1. **Data Preparation**:
   - Collect and annotate images for each food class
   - Split into training and validation sets
   - Create data.yaml configuration file

2. **Training the Model**:
   ```python
   from ultralytics import YOLO
   
   # Initialize with pre-trained weights
   model = YOLO("yolov11n.pt")  
   
   # Train the model
   results = model.train(
       data="path/to/data.yaml",
       epochs=100,
       imgsz=640
   )
   ```

3. **Validation**:
   ```python
   metrics = model.val()
   print(f"mAP50-95: {metrics.box.map}")
   print(f"mAP50: {metrics.box.map50}")
   ```

## Data Annotation

1. **Collect Images**: Gather diverse images of each food class
2. **Annotation Tool**: Use tools like [Roboflow](https://roboflow.com/) or [CVAT](https://cvat.org/) for bounding box annotations
3. **Export Format**: Export annotations in YOLO format (text files with normalized coordinates)
4. **Directory Structure**:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```

## Inference

### Using the Model

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("best-6.pt")

# Run inference on an image
results = model("path/to/image.jpg")

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        class_name = model.names[cls]
        print(f"Detected {class_name} with confidence {conf:.2f}")
```

## Nutrition Data

The application uses a nutritional database (`nutrition_table.csv`) containing information about various food items, including:
- Calories
- Protein
- Fat
- Carbohydrates
- Fiber
- Vitamins and minerals

## Project Structure

```
food-and-calorie-estimation/
├── app.py                    # Main Streamlit application
├── best-6.pt                 # Trained YOLO model weights
├── nutrition_table.csv       # Nutritional information database
├── requirements.txt          # Project dependencies
├── temp/                     # Temporary files for application
└── README.md                 # This file
```

## Future Improvements

- Support for more food classes
- Portion size estimation
- Mobile application
- Personalized dietary recommendations
- Integration with fitness trackers

## License

[MIT License](LICENSE)

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/) 