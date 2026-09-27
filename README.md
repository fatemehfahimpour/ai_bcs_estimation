# Dairy Cow Body Condition Score (BCS) Estimation

This project implements a two-stage deep learning pipeline for estimating the **Body Condition Score (BCS)** of dairy cows from images.

The final inference pipeline consists of two main stages:

1. **YOLO11m** detects the relevant anatomical region of the cow.
2. **Ordinal ResNet18** predicts the BCS from the detected and cropped region.

The supported BCS values are:

- 3.25
- 3.50
- 3.75
- 4.00
- 4.25

The complete pipeline is:

```text
Input Cow Image
       |
       v
YOLO11m Anatomical Region Detector
       |
       v
Detected / Cropped Region
       |
       v
Image Preprocessing
       |
       v
Ordinal ResNet18
       |
       v
Predicted BCS
```

---

## 1. Project Structure

The main project structure is:

```text
BCS_2/
│
├── anatomical_region_ditector/
│   ├── evaluation_results/
│   ├── runs/
│   │   └── detect/
│   │       └── anatomical_region_yolo11m/
│   │           └── weights/
│   │               ├── best.pt
│   │               └── ...
│   ├── create_cropped_dataset.py
│   ├── crop_check.py
│   ├── data_detector.yaml
│   ├── detect_train_output.txt
│   ├── evaluate_detector.py
│   ├── evaluate_detector_output.txt
│   ├── prepare_detector_dataset.py
│   ├── train_detector.py
│   └── yolo11m.pt
│
├── data/
│   └── ...
│
├── dataset/
│   └── ...
│
├── inference/
│   ├── inference.py
│   └── gui_app.py
│
├── ReNet18_ordinal_classification_model/
│   ├── find_best_ordinal_model.py
│   ├── find_best_ordinal_model_2.py
│   ├── ordinal_best_model_test.py
│   ├── ordinal_dataset_dataloader.py
│   ├── ordinal_model.py
│   ├── ordinal_trainer.py
│   ├── model_results/
│   ├── model_results_round2/
│   │   ├── best_ordinal_resnet18.pth
│   │   ├── ordinal_training_result_2.json
│   │   └── plots/
│   └── ...
│
├── ResNet18_classification_model/
│   ├── ResNet_best_model_test.py
│   ├── ResNet_hyperparameter_search.py
│   ├── ResNet_model.py
│   └── ResNet_trainer.py
│
├── runs/
│   └── ...
│
├── .gitignore
├── report.docx
├── requirements.txt
├── test_env.py
└── README.md
```

### Main Directories

### `anatomical_region_ditector/`

Contains the YOLO11m-based anatomical region detection pipeline.

Important files:

- `prepare_detector_dataset.py` prepares the detector dataset.
- `data_detector.yaml` contains the YOLO dataset configuration.
- `train_detector.py` trains the YOLO11m detector.
- `evaluate_detector.py` evaluates the trained detector.
- `create_cropped_dataset.py` creates cropped images using the trained detector.
- `crop_check.py` is used to inspect cropped images.
- `yolo11m.pt` is the pretrained YOLO11m starting weight.
- `runs/detect/anatomical_region_yolo11m/weights/best.pt` is the trained YOLO11m checkpoint used by the final inference pipeline.

### `data/`

Contains shared data loading and image preprocessing utilities used by the classification pipeline.

### `dataset/`

Contains the original dataset and related dataset files used during the project.

### `inference/`

Contains the final inference implementation.

- `inference.py` contains the `BCSInferenceEngine`.
- `gui_app.py` provides a graphical interface for running inference on a new image.

### `ReNet18_ordinal_classification_model/`

Contains the final ordinal classification implementation.

Important files:

- `ordinal_model.py` defines the `OrdinalResNet18` model.
- `ordinal_dataset_dataloader.py` creates the ordinal datasets and DataLoaders.
- `ordinal_trainer.py` contains the training logic.
- `find_best_ordinal_model.py` contains the first hyperparameter search.
- `find_best_ordinal_model_2.py` contains the final hyperparameter search.
- `ordinal_best_model_test.py` evaluates the final model.
- `model_results_round2/best_ordinal_resnet18.pth` is the final trained Ordinal ResNet18 checkpoint.

### `ResNet18_classification_model/`

Contains the earlier standard 5-class ResNet18 classification experiments.

This model was used as an initial baseline before switching to ordinal classification.

---

# 2. Requirements

The project uses Python and several machine learning and computer vision libraries.

The required dependencies are listed in:

```text
requirements.txt
```

The main dependencies include:

- PyTorch
- Torchvision
- Ultralytics
- NumPy
- Pandas
- Pillow
- Scikit-learn
- Matplotlib
- Seaborn

---

# 3. Installation

## 3.1. Clone the Repository

Clone the project repository:

```bash
git clone <REPOSITORY_URL>
```

Move into the project directory:

```bash
cd BCS_2
```

## 3.2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

## 3.3. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

If the project is being used with an NVIDIA GPU, make sure that the installed PyTorch version is compatible with the CUDA environment of the system.

## 3.4. Check the Environment

The project contains:

```text
test_env.py
```

Run:

```bash
python test_env.py
```

This checks the Python environment and reports information such as:

- Python version
- PyTorch version
- Python executable
- CUDA availability
- GPU availability

---

# 4. Model Weights

The final inference pipeline requires two trained models:

1. YOLO11m for anatomical region detection.
2. Ordinal ResNet18 for BCS prediction.

## 4.1. YOLO11m

The original pretrained YOLO11m weight is:

```text
anatomical_region_ditector/yolo11m.pt
```

This file is used as the starting point for training the detector.

After training, the best trained detector checkpoint is stored at:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

The final inference pipeline uses:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

Therefore, `best.pt` is the important trained YOLO weight required for final inference.

## 4.2. Ordinal ResNet18

The final trained Ordinal ResNet18 checkpoint is:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

This checkpoint contains the trained model parameters and configuration information required to reconstruct the trained model.

## 4.3. Required Weights

Before running the final inference pipeline, make sure these files exist:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt

ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

If these files are not included in the Git repository because of their size, download them from the links provided below:

- YOLO11m trained weights: `<YOLO_WEIGHT_DOWNLOAD_LINK>`
- Ordinal ResNet18 trained weights: `<ORDINAL_RESNET18_WEIGHT_DOWNLOAD_LINK>`

After downloading the files, place them in the exact paths shown above.

---

# 5. Dataset

The project uses images of dairy cows together with their Body Condition Scores.

The supported BCS values are:

```text
3.25
3.50
3.75
4.00
4.25
```

The dataset metadata contains information such as:

```text
file_name
bcs
prefix
cow_id
image_id
cow_group_id
path
```

The most important identifier for splitting the data is:

```text
cow_group_id
```

This identifier is used to group images belonging to the same cow.

---

# 6. Dataset Splitting

The final dataset is divided into:

| Subset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 25% |
| Test | 15% |

The split is performed at the **cow-group level**, not at the individual image level.

This is important because one cow can have multiple images.

If images belonging to the same cow were placed in both training and test sets, the model could learn cow-specific visual characteristics. This would cause data leakage and could make the evaluation results unreliable.

Therefore, images belonging to the same `cow_group_id` are kept in only one subset.

The split files are:

```text
data/meta_data/splits/
├── train.csv
├── val.csv
└── test.csv
```

The split files are already generated and are directly used by the training and evaluation pipelines.

### Important

If the dataset is changed and the splits need to be recreated, the split must still be performed at the cow-group level.

Do not perform a simple random image-level split.

The split should also preserve the BCS distribution as closely as possible between the subsets.

---

# 7. BCS Classification

The BCS values are ordered:

```text
3.25 < 3.50 < 3.75 < 4.00 < 4.25
```

The project initially used standard 5-class ResNet18 classification.

In the standard classification approach:

```text
3.25 -> 0
3.50 -> 1
3.75 -> 2
4.00 -> 3
4.25 -> 4
```

However, treating these values as completely independent classes ignores their natural ordering.

The final model therefore uses **ordinal classification**.

The ordinal model predicts four thresholds:

```text
BCS > 3.25
BCS > 3.50
BCS > 3.75
BCS > 4.00
```

The corresponding targets are:

```text
3.25 -> [0, 0, 0, 0]
3.50 -> [1, 0, 0, 0]
3.75 -> [1, 1, 0, 0]
4.00 -> [1, 1, 1, 0]
4.25 -> [1, 1, 1, 1]
```

The Ordinal ResNet18 therefore has four output neurons and is trained using:

```text
BCEWithLogitsLoss
```

During inference, the four sigmoid probabilities are thresholded at `0.5`.

The number of thresholds that are passed determines the final BCS class.

---

# 8. YOLO11m Detector

The first stage of the final pipeline is anatomical region detection.

The detector is based on:

```text
YOLO11m
```

The detector identifies the relevant anatomical region required for BCS estimation.

## Detector Dataset Preparation

The detector dataset can be prepared using:

```bash
python anatomical_region_ditector/prepare_detector_dataset.py
```

The preparation process uses the original images, dataset split information, and the corresponding bounding-box annotations.

The generated detector dataset follows the YOLO format.

The detector configuration is:

```text
anatomical_region_ditector/data_detector.yaml
```

## Detector Training

The training code is:

```text
anatomical_region_ditector/train_detector.py
```

The detector is trained using the pretrained:

```text
anatomical_region_ditector/yolo11m.pt
```

The trained model is saved under:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/
```

The best checkpoint is:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

## Detector Evaluation

Evaluate the trained detector using:

```bash
python anatomical_region_ditector/evaluate_detector.py
```

The evaluation results are stored under:

```text
anatomical_region_ditector/evaluation_results/
```

---

# 9. Creating Cropped Images

The trained YOLO detector can also be used to create a cropped version of the dataset.

Run:

```bash
python anatomical_region_ditector/create_cropped_dataset.py
```

The script uses the trained detector:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

and generates cropped images from the original dataset.

The cropping process selects the detected region and saves the resulting images for use in classification experiments.

The `crop_check.py` script can be used to inspect the generated crops:

```bash
python anatomical_region_ditector/crop_check.py
```

---

# 10. Ordinal ResNet18 Training

The final BCS classification model is:

```text
Ordinal ResNet18
```

The model definition is located at:

```text
ReNet18_ordinal_classification_model/ordinal_model.py
```

The DataLoader is implemented in:

```text
ReNet18_ordinal_classification_model/ordinal_dataset_dataloader.py
```

The training logic is implemented in:

```text
ReNet18_ordinal_classification_model/ordinal_trainer.py
```

The final hyperparameter search is:

```text
ReNet18_ordinal_classification_model/find_best_ordinal_model_2.py
```

The final search evaluates different combinations of:

- trainable layers
- optimizer
- learning rate

The final trained model is saved as:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

The search results are stored in:

```text
ReNet18_ordinal_classification_model/model_results_round2/ordinal_training_result_2.json
```

and training plots are stored in:

```text
ReNet18_ordinal_classification_model/model_results_round2/plots/
```

---

# 11. Final Inference Pipeline

The final inference implementation is located in:

```text
inference/inference.py
```

The main class is:

```text
BCSInferenceEngine
```

The engine loads both trained models:

```text
YOLO11m best.pt
        +
Ordinal ResNet18 best_ordinal_resnet18.pth
```

The complete inference process is:

```text
Full Cow Image
       |
       v
YOLO11m
       |
       v
Anatomical Region Detection
       |
       v
Crop
       |
       v
Image Preprocessing
       |
       v
Ordinal ResNet18
       |
       v
Predicted BCS
```

The detector uses the trained YOLO model to locate the relevant anatomical region.

The detected region is then cropped and passed through the same type of preprocessing used by the classification model.

Finally, the Ordinal ResNet18 predicts the BCS.

---

# 12. Running Inference on a New Image

There are two ways to use the final inference system.

## 12.1. Graphical Interface

The easiest way to run the complete pipeline is through:

```text
inference/gui_app.py
```

From the project root, run:

```bash
python inference/gui_app.py
```

The GUI allows the user to:

1. Select a cow image.
2. Run the YOLO11m detector.
3. Crop the detected anatomical region.
4. Run the Ordinal ResNet18 classifier.
5. Display the predicted BCS.
6. Display the detection and prediction information.

### Step-by-step

After starting the application:

```bash
python inference/gui_app.py
```

select an input image using the image selection button.

Then run the prediction.

The application performs:

```text
Selected Image
      |
      v
YOLO11m
      |
      v
Detected Region
      |
      v
Crop
      |
      v
Ordinal ResNet18
      |
      v
BCS Prediction
```

---

# 13. Using `inference.py` Directly

The file:

```text
inference/inference.py
```

contains the inference engine.

The current implementation does not provide a command-line interface using `argparse`.

Therefore, the following is **not** the correct way to run the current implementation:

```bash
python inference/inference.py image.jpg
```

Instead, the engine can be called directly from Python.

For example:

```python
from inference.inference import BCSInferenceEngine

engine = BCSInferenceEngine()

result = engine.predict_full_pipeline("path/to/cow_image.jpg")

print("Predicted BCS:", result["predicted_bcs"])
print("Detector confidence:", result.get("detector_conf"))
print("Prediction certainty:", result["confidence"])
print("Threshold probabilities:", result["threshold_probabilities"])
```

For example:

```python
from inference.inference import BCSInferenceEngine

engine = BCSInferenceEngine()

result = engine.predict_full_pipeline("sample_images/cow.jpg")

print("Predicted BCS:", result["predicted_bcs"])
```

The returned result contains information such as:

```text
detected
bbox
detector_conf
crop_image
predicted_bcs
confidence
threshold_probabilities
```

---

# 14. `predict_crop()` vs `predict_full_pipeline()`

The inference engine provides two main prediction methods.

## `predict_crop()`

Use this method when the anatomical region has already been cropped.

```text
Cropped Image
      |
      v
Preprocessing
      |
      v
Ordinal ResNet18
      |
      v
BCS Prediction
```

## `predict_full_pipeline()`

Use this method when the input is a normal full cow image.

```text
Full Image
      |
      v
YOLO11m
      |
      v
Anatomical Region
      |
      v
Crop
      |
      v
Ordinal ResNet18
      |
      v
BCS Prediction
```

For a new normal cow image, `predict_full_pipeline()` should be used.

---

# 15. Inference Output

The final prediction is one of:

```text
3.25
3.50
3.75
4.00
4.25
```

The ordinal model produces four threshold probabilities:

```text
P(BCS > 3.25)
P(BCS > 3.50)
P(BCS > 3.75)
P(BCS > 4.00)
```

For example:

```text
[0.91, 0.84, 0.32, 0.11]
```

means that the first two thresholds are passed.

Therefore:

```text
Class index = 2
Predicted BCS = 3.75
```

---

# 16. Test Evaluation

The final Ordinal ResNet18 model can be evaluated using:

```text
ReNet18_ordinal_classification_model/ordinal_best_model_test.py
```

Run:

```bash
python ReNet18_ordinal_classification_model/ordinal_best_model_test.py
```

The script loads:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

and evaluates it using the test split.

The test results are stored under:

```text
test_results/
```

The generated files include:

```text
test_results/
├── test_predictions.csv
├── test_metrics.json
├── classification_report.txt
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── actual_vs_predicted.png
├── bcs_distribution.png
└── error_distribution.png
```

The evaluation includes:

- Exact accuracy
- Accuracy with `±0.25` tolerance
- MAE
- RMSE
- Maximum error
- Mean absolute error
- Classification report
- Confusion matrix
- Normalized confusion matrix
- Actual vs. predicted plot
- BCS distribution comparison
- Error distribution

The tolerance criterion is:

```text
|Actual BCS - Predicted BCS| <= 0.25
```

Tolerance affects the definition of correctness for the tolerance-based accuracy. It does not change the raw MAE or RMSE.

---

# 17. Continuing the Project

A new developer/researcher continuing this project should understand the following dependency chain:

```text
Original Dataset
       |
       v
Metadata and Cow-Level Splits
       |
       +---------------------------+
       |                           |
       v                           v
Detector Dataset              Classification Dataset
       |                           |
       v                           v
YOLO11m Training              Ordinal ResNet18 Training
       |                           |
       v                           v
best.pt                       best_ordinal_resnet18.pth
       |                           |
       +-------------+-------------+
                     |
                     v
              Final Inference
                     |
                     v
                Predicted BCS
```

## If the goal is only to use the trained system

Do not retrain the models.

Make sure these two files exist:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt

ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

Then run:

```bash
python inference/gui_app.py
```

## If the YOLO detector needs to be retrained

Follow this order:

```text
1. Check the original dataset.
2. Check the dataset split files.
3. Prepare the detector dataset.
4. Check data_detector.yaml.
5. Train YOLO11m.
6. Evaluate the detector.
7. Verify best.pt.
```

Commands:

```bash
python anatomical_region_ditector/prepare_detector_dataset.py
python anatomical_region_ditector/train_detector.py
python anatomical_region_ditector/evaluate_detector.py
```

## If the Ordinal ResNet18 needs to be retrained

Follow this order:

```text
1. Check the train/validation/test split files.
2. Check data/preprocess.py.
3. Check ordinal_dataset_dataloader.py.
4. Run the hyperparameter search.
5. Check the generated checkpoint.
6. Evaluate the final model.
```

Commands:

```bash
python ReNet18_ordinal_classification_model/find_best_ordinal_model_2.py
python ReNet18_ordinal_classification_model/ordinal_best_model_test.py
```

The final classifier checkpoint will be:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

---

# 18. Important File Dependencies

The following files are important for the final pipeline.

### YOLO checkpoint

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

Used by:

```text
inference/inference.py
anatomical_region_ditector/evaluate_detector.py
anatomical_region_ditector/create_cropped_dataset.py
```

### Ordinal ResNet18 checkpoint

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

Used by:

```text
inference/inference.py
ReNet18_ordinal_classification_model/ordinal_best_model_test.py
```

### Dataset splits

```text
data/meta_data/splits/train.csv
data/meta_data/splits/val.csv
data/meta_data/splits/test.csv
```

These files are used by the classification pipeline and detector dataset preparation.

### Preprocessing

```text
data/preprocess.py
```

This file contains shared preprocessing utilities used by the classification data pipeline and inference.

Changes to preprocessing can affect model performance. If preprocessing is changed, the trained model should be evaluated again.

---

# 19. Common Problems

## Model checkpoint not found

If the Ordinal ResNet18 checkpoint cannot be found, check:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

If the YOLO checkpoint cannot be found, check:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

## `ModuleNotFoundError`

Make sure the command is being executed from the project root:

```text
BCS_2/
```

Also activate the virtual environment:

```bash
.venv\Scripts\activate
```

and install dependencies:

```bash
pip install -r requirements.txt
```

## CUDA is unavailable

The inference engine automatically uses CUDA when it is available and otherwise falls back to CPU.

CPU inference is supported but may be slower.

## YOLO does not detect the region

If the YOLO detector does not detect the required region, the current inference implementation can fall back to using the full image as the classifier input.

---

# 20. Quick Start

For someone who only wants to run the existing trained system:

### Step 1 — Clone the repository

```bash
git clone <REPOSITORY_URL>
cd BCS_2
```

### Step 2 — Create the virtual environment

```bash
python -m venv .venv
```

### Step 3 — Activate the virtual environment

```bash
.venv\Scripts\activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Check the environment

```bash
python test_env.py
```

### Step 6 — Check the trained weights

Make sure these files exist:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt

ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

### Step 7 — Start the inference application

```bash
python inference/gui_app.py
```

### Step 8 — Select a cow image

Use the image selection button in the GUI.

### Step 9 — Run prediction

Run the BCS prediction.

The complete pipeline will be:

```text
Input Image
     |
     v
YOLO11m
     |
     v
Anatomical Region Detection
     |
     v
Crop + Preprocessing
     |
     v
Ordinal ResNet18
     |
     v
Predicted BCS
```

---

# 21. Final Pipeline Summary

The final system consists of two trained deep learning models:

```text
YOLO11m
   +
Ordinal ResNet18
```

The first model detects the relevant anatomical region of the cow.

The second model predicts the ordered BCS value from that region.

The complete system is:

```text
                    +----------------------+
                    |      Cow Image       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |       YOLO11m        |
                    | Anatomical Detection |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Region Cropping    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Preprocessing      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Ordinal ResNet18   |
                    |    4 Thresholds      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    Predicted BCS      |
                    | 3.25 / 3.50 / 3.75  |
                    | 4.00 / 4.25         |
                    +----------------------+
```

For normal use, the main entry point is:

```bash
python inference/gui_app.py
```

The main inference engine is:

```text
inference/inference.py
```

The final trained model weights are:

```text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

and:

```text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```