# ModelComparison
This repository provides code for comparing face detection models (Haar Cascade, HOG, YOLO) on custom or public image datasets. The project is designed to benchmark classic and modern face detectors, visualize their results, and compare performance using metrics such as accuracy, precision, recall, F1-score, and inference speed.

## Features

## Dataset support: 
Works with datasets organized by folders (one folder per person/class).

## Model Comparison: 
Easily run Haar Cascade, HOG, and YOLO face detection.

## Metrics & Visualization: 
Plots confusion matrices, accuracy, loss, and per-model comparison graphs.

## Performance Logging: 
Records inference time, number of detections, and detection quality for each model.

## Easy configuration: 
Switch between models with a single line; add more models as needed.

## Example notebooks: 
Ready-to-use Jupyter notebooks for quick experimentation.

## Contents

### CompareModels.ipynb: 
A Jupyter Notebook that loads datasets, trains multiple machine learning models, and compares their performance using metrics such as accuracy, precision, recall, and F1-score.

### Images/:
 A directory containing visualizations generated during the model comparison process, including confusion matrices and performance plots.


## Requirements
To run the notebook, ensure you have the following Python packages installed:

numpy

pandas

matplotlib

scikit-learn

seaborn

You can install them using pip:
``` 
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Usage
### Clone the repository
```
git clone https://github.com/MuhammedZhumali/ModelComparison.git
cd ModelComparison
```
### Prepare your dataset
Place your images in the dataset directory, with subfolders for each class/person.

### Run the notebook
```
jupyter notebook CompareModels.ipynb
```
Follow the cells to select models, run detection, and generate comparison plots.

## Results & Metrics

### Accuracy, Precision, Recall, F1-score

### Inference Time per image

### ROC Curves and Confusion Matrix

### Loss & Accuracy plots per model


## Contact
For questions or collaboration:

Author: Muhammed Zhumali

Email: zhumalimuhammed@gmail.com

## Acknowledgments

### Ultralytics YOLO repository: https://github.com/ultralytics/ultralytics

### Dlib face recognition: http://dlib.net/

### OpenCV documentation: https://docs.opencv.org/


# ^^
