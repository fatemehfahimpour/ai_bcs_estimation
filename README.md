# AI-Based Cow Body Condition Score (BCS) Estimation

An end-to-end deep learning project for estimating the **Body Condition
Score (BCS)** of cows from images.

The project follows a **two-stage computer vision pipeline**:

1.  **Anatomical-region detection:** a YOLO11m object detector
    identifies the relevant anatomical region of the cow.
2.  **BCS estimation:** the detected/cropped region is passed to a
    ResNet18-based BCS classifier. The main inference pipeline uses an
    **Ordinal ResNet18**, which models the ordered nature of BCS
    classes.

The repository also contains a conventional ResNet18 classification
pipeline, dataset analysis and cleaning utilities, subject-level
train/validation/test splitting, detector evaluation tools,
model-selection scripts, and a Tkinter-based inference GUI.

------------------------------------------------------------------------

## Table of Contents

- [Project Overview](#project-overview)
- [BCS Classes](#bcs-classes)
- [Main Features](#main-features)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Files Generated Locally](#files-generated-locally)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Dataset Preparation](#dataset-preparation)
- [Training the Anatomical-Region Detector](#training-the-anatomical-region-detector)
- [Creating the Cropped Dataset](#creating-the-cropped-dataset)
- [Classification Preprocessing](#classification-preprocessing)
- [Training the Standard ResNet18 Classifier](#training-the-standard-resnet18-classifier)
- [Training the Ordinal ResNet18 Classifier](#training-the-ordinal-resnet18-classifier)
- [Evaluating the Models](#evaluating-the-models)
- [Running Inference](#running-inference)
- [Running the GUI](#running-the-gui)
- [Environment Verification](#environment-verification)
- [Reproducibility Notes](#reproducibility-notes)
- [Important Implementation Details](#important-implementation-details)
- [Troubleshooting](#troubleshooting)
- [Team Members](#team-members)
- [Suggested Future Improvements](#suggested-future-improvements)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

------------------------------------------------------------------------

## Project Overview

The goal of this project is to estimate cow Body Condition Score from
visual information.

The implemented BCS classes are:

    Class Index    BCS
  ------------- ------
              0   3.25
              1   3.50
              2   3.75
              3   4.00
              4   4.25

The classification preprocessing uses ImageNet normalization and resizes
inputs to **224 × 224** while preserving the original aspect ratio
through gray padding rather than directly cropping the image.

The ordinal model represents the five BCS classes using **four
cumulative ordinal thresholds**.

------------------------------------------------------------------------

## Main Features

-   Dataset metadata generation and exploratory analysis
-   Image-quality checking and cleaning
-   Subject-level train/validation/test splitting to reduce cow-level
    data leakage
-   YOLO-format object-detection dataset generation from Pascal VOC XML
    annotations
-   YOLO11m anatomical-region detection
-   Automatic generation of a cropped dataset using the trained detector
-   Standard ResNet18 BCS classification
-   Ordinal ResNet18 BCS classification
-   Hyperparameter search for both classification approaches
-   Test-set evaluation and visualization
-   Two-stage end-to-end inference
-   Interactive Tkinter GUI
-   CPU/GPU device detection through PyTorch
-   Automatic saving of training checkpoints and evaluation artifacts

------------------------------------------------------------------------

## Pipeline

``` text
Raw Dataset
│
├── Images
└── Pascal VOC XML annotations
        │
        ▼
Dataset Analysis
        │
        ▼
Image Quality Cleaning
        │
        ▼
Subject-Level Split
        │
        ├── train.csv
        ├── val.csv
        └── test.csv
        │
        ├───────────────────────────────┐
        ▼                               ▼
YOLO Dataset Preparation          Classification DataLoader
        │                               │
        ▼                               ▼
Anatomical Region Detection      ResNet18 / Ordinal ResNet18
        │                               │
        ▼                               ▼
Trained YOLO11m                 Trained BCS Classifier
        │                               │
        └──────────────┬────────────────┘
                       ▼
              Two-Stage Inference
                       │
                       ▼
                BCS Prediction
                       │
                       ▼
                  GUI / API
```

------------------------------------------------------------------------

## Repository Structure

``` text
ai_bcs_estimation/
│
├── anatomical_region_ditector/
│   ├── evaluation_results/              # Detector evaluation outputs
│   │   └── worst_predictions/
│   ├── runs/                            # YOLO training/validation runs
│   │   └── detect/
│   │       └── anatomical_region_yolo11m/
│   │           └── weights/
│   │               ├── best.pt
│   │               └── last.pt
│   ├── create_cropped_dataset.py        # Create BCS classification crops
│   ├── crop_check.py                    # Crop inspection utility
│   ├── data_detector.yaml               # YOLO dataset configuration
│   ├── evaluate_detector.py             # Detector evaluation
│   ├── prepare_detector_dataset.py      # VOC/XML → YOLO conversion
│   ├── train_detector.py                # YOLO11m training
│   └── yolo11m.pt                       # Base YOLO11m weights
│
├── cropped_dataset/                     # GENERATED / LOCAL
│   ├── 3.25/
│   ├── 3.5/
│   ├── 3.75/
│   ├── 4.0/
│   └── 4.25/
│
├── data/
│   ├── augmentation_preview/            # Preprocessing visualizations
│   ├── meta_data/
│   │   └── splits/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   ├── clean_data.py                    # Image quality filtering
│   ├── dataset_analysis.py              # Dataset analysis/metadata
│   ├── preprocess.py                    # Transforms and PyTorch DataLoaders
│   └── split.py                         # Subject-level dataset splitting
│
├── dataset/                             # RAW DATASET / LOCAL
│   ├── 3.25/
│   ├── 3.5/
│   ├── 3.75/
│   ├── 4.0/
│   └── 4.25/
│
├── detector_dataset/                    # GENERATED / LOCAL
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── inference/
│   ├── gui_app.py                       # Tkinter GUI
│   └── inference.py                     # Two-stage inference engine
│
├── ReNet18_ordinal_classification_model/
│   ├── model_results/                   # First-round results
│   ├── model_results_round2/            # Refined ordinal search/results
│   ├── test_results/                    # Ordinal test outputs
│   ├── find_best_ordinal_model.py
│   ├── find_best_ordinal_model_2.py
│   ├── ordinal_best_model_test.py
│   ├── ordinal_dataset_dataloader.py
│   ├── ordinal_model.py
│   └── ordinal_trainer.py
│
├── ResNet18_classification_model/
│   ├── model_results/
│   ├── test_results/
│   ├── ResNet_best_model_test.py
│   ├── ResNet_hyperparameter_search.py
│   ├── ResNet_model.py
│   └── ResNet_trainer.py
│
├── runs/                                # Additional YOLO validation artifacts
│
├── .gitignore
├── README.md
├── requirements.txt
└── test_env.py
```

> **Note:** Some generated/output directories shown above may not exist
> immediately after cloning. They are created by the corresponding
> scripts.

------------------------------------------------------------------------

## Files Generated Locally

The repository's `.gitignore` excludes the raw dataset and several
generated/local directories. In particular, the following should be
recreated on the user's machine rather than expected to be downloaded
from GitHub:

``` text
dataset/
detector_dataset/
cropped_dataset/
.idea/
venv/
.venv/
__pycache__/
*.pyc
```

This means a fresh clone should be prepared using the instructions below
before running the full training/inference pipeline.

------------------------------------------------------------------------

## Requirements

The project uses Python and the following major libraries:

-   PyTorch
-   Torchvision
-   Ultralytics
-   NumPy
-   Pandas
-   Pillow
-   OpenCV
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   tqdm

Tkinter is also required for the graphical application.

### Recommended installation

If `requirements.txt` is present in the repository:

``` bash
pip install -r requirements.txt
```

For a clean environment, using a virtual environment is recommended.

### Windows

``` powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The supplied source code does not pin a single Python version inside
> the scripts. The project's `requirements.txt` should therefore be
> treated as the source of truth for the exact package versions used by
> the project.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/fatemehfahimpour/ai_bcs_estimation.git
cd ai_bcs_estimation
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` powershell
.venv\Scripts\activate
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the environment

From the project root:

``` bash
python test_env.py
```

The script reports:

-   Operating system
-   Python version
-   Python interpreter path
-   PyTorch version
-   Whether CUDA/GPU or CPU will be used

------------------------------------------------------------------------

## Dataset

The project expects the raw dataset to be available locally under:

``` text
dataset/
```

The classification workflow uses BCS folders corresponding to:

``` text
dataset/
├── 3.25/
├── 3.5/
├── 3.75/
├── 4.0/
└── 4.25/
```

For anatomical-region detection, the dataset images must also have
corresponding **Pascal VOC XML annotations**. The detector preparation
script searches for image/XML pairs and converts the bounding boxes to
YOLO format.

### Dataset link


``` text
Dataset source:
https://www.scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c
```



------------------------------------------------------------------------

## Dataset Preparation

The recommended order is:

### Step 1 --- Dataset analysis

From the `data/` directory:

``` bash
cd data
python dataset_analysis.py
```

This analyzes the BCS directory structure and extracts metadata such as
cow IDs, image information, locations, and BCS labels.

### Step 2 --- Image-quality cleaning

Still inside `data/`:

``` bash
python clean_data.py
```

The cleaning module checks:

-   Whether images are readable
-   Extremely dark images
-   Extremely bright images
-   Image blurriness using Laplacian variance

The current thresholds are defined directly in `clean_data.py`.

### Step 3 --- Subject-level splitting

``` bash
python split.py
```

The splitting procedure works at the cow/subject level rather than
simply splitting individual images. The current configured proportions
are:

``` text
Train:      60%
Validation: 25%
Test:       15%
```

The generated files are:

``` text
data/meta_data/splits/
├── train.csv
├── val.csv
└── test.csv
```

A fixed random seed of `42` is used by the splitting script.

------------------------------------------------------------------------

## Training the Anatomical-Region Detector

The detector is based on **YOLO11m**.

### Step 1 --- Prepare the YOLO dataset

The preparation script reads:

-   the raw dataset
-   the train/validation/test metadata
-   Pascal VOC XML annotations

and creates:

``` text
detector_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Run:

``` bash
cd anatomical_region_ditector
python prepare_detector_dataset.py
```

This also generates:

``` text
data_detector.yaml
```

### Step 2 --- Train YOLO11m

``` bash
python train_detector.py
```

The current configuration in the script is:

``` text
Model:       YOLO11m
Epochs:      100
Image size:  640
Batch size:  16
Early stop:  25 epochs patience
```

The script automatically selects CUDA when available according to its
device configuration and otherwise falls back to CPU.

The main trained model is expected at:

``` text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

### Step 3 --- Evaluate the detector

``` bash
python evaluate_detector.py
```

The evaluation creates artifacts such as:

``` text
anatomical_region_ditector/evaluation_results/
├── test_set_iou_report.csv
└── worst_predictions/
```

The evaluation script also exports visual examples of
low-IoU/worst-performing detections for inspection.

------------------------------------------------------------------------

## Creating the Cropped Dataset

After the detector has been trained successfully, create the cropped BCS
dataset:

``` bash
python anatomical_region_ditector/create_cropped_dataset.py
```

The script uses the trained detector to locate the relevant anatomical
region and saves the resulting crops under:

``` text
cropped_dataset/
├── 3.25/
├── 3.5/
├── 3.75/
├── 4.0/
└── 4.25/
```

The crop-generation script uses the detector weights located at:

``` text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

------------------------------------------------------------------------

## Classification Preprocessing

The classification preprocessing is implemented in:

``` text
data/preprocess.py
```

### Training preprocessing

The training pipeline includes:

1.  Aspect-ratio-preserving square padding
2.  Random horizontal flip
3.  Safe random rotation
4.  Additional square padding
5.  Resize to 224 × 224
6.  Color jitter
7.  Conversion to tensor
8.  ImageNet normalization

### Validation/test preprocessing

Validation and test data use:

1.  Aspect-ratio-preserving square padding
2.  Resize to 224 × 224
3.  Conversion to tensor
4.  ImageNet normalization

The project intentionally avoids direct cropping in this preprocessing
pipeline.

------------------------------------------------------------------------

## Training the Standard ResNet18 Classifier

The conventional classification implementation is located in:

``` text
ResNet18_classification_model/
```

Run the hyperparameter search from that directory:

``` bash
cd ResNet18_classification_model
python ResNet_hyperparameter_search.py
```

The current search explores:

-   Trainable layers:
    -   `layer4_fc`
    -   `layer3_layer4_fc`
-   Optimizer:
    -   Adam
-   Learning rates:
    -   `1e-5`
    -   `3e-5`
    -   `1e-4`
    -   `3e-4`
    -   `1e-3`
-   Maximum epochs: `100`
-   Early-stopping patience: `20`

The best model is saved as:

``` text
ResNet18_classification_model/model_results/best_resnet18.pth
```

The hyperparameter-search results are saved as:

``` text
ResNet18_classification_model/model_results/resnet_hyperparameter_results.json
```

------------------------------------------------------------------------

## Training the Ordinal ResNet18 Classifier

The main inference pipeline uses the ordinal model because BCS values
are ordered.

The ordinal implementation is located in:

``` text
ReNet18_ordinal_classification_model/
```

> The directory name `ReNet18_ordinal_classification_model` is retained
> exactly as it appears in the project source.

### Round 1

``` bash
cd ReNet18_ordinal_classification_model
python find_best_ordinal_model.py
```

### Round 2

The refined search is:

``` bash
python find_best_ordinal_model_2.py
```

The current Round 2 configuration explores:

``` text
Trainable layers:
- layer4_fc
- layer3_layer4_fc
- layer2_layer3_layer4_fc

Optimizers:
- Adam
- Momentum

Adam learning rates:
- 5e-6
- 1e-5
- 3e-5

Momentum learning rates:
- 3e-4
- 1e-3
- 3e-3

Epochs:       100
Patience:     20
Min delta:    0.001
Backbone LR ratio: 0.1
```

The best Round 2 checkpoint is expected at:

``` text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

Additional training results and plots are stored under:

``` text
ReNet18_ordinal_classification_model/model_results_round2/
```

------------------------------------------------------------------------

## Evaluating the Models

### Standard ResNet18

From:

``` text
ResNet18_classification_model/
```

run:

``` bash
python ResNet_best_model_test.py
```

The test script generates metrics and visualizations such as:

``` text
test_results/
├── confusion_matrix.png
├── normalized_confusion_matrix.png
├── class_distribution.png
├── error_distribution.png
├── actual_vs_predicted.png
├── test_metrics.json
├── classification_report.txt
└── test_predictions.csv
```

### Ordinal ResNet18

From:

``` text
ReNet18_ordinal_classification_model/
```

run:

``` bash
python ordinal_best_model_test.py
```

The test artifacts are saved under:

``` text
test_results/
```

and include confusion matrices, predictions, distributions, metrics, and
classification reports.

------------------------------------------------------------------------

## Running Inference

The two-stage inference engine is implemented in:

``` text
inference/inference.py
```

The default pipeline expects:

``` text
Classifier:
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth

Detector:
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

The engine automatically uses CUDA when it is available; otherwise it
uses CPU.

### Stage 1

The YOLO detector searches for the anatomical region.

### Stage 2

The detected region is cropped and passed to the ordinal BCS classifier.

### No-detection fallback

If the detector does not find a bounding box, the inference engine falls
back to using the full input image for BCS prediction.

### Example

A Python script can use the engine as follows:

``` python
from inference.inference import BCSInferenceEngine

engine = BCSInferenceEngine()

result = engine.predict_full_pipeline("path/to/cow_image.jpg")

print("Detected:", result["detected"])
print("Predicted BCS:", result["predicted_bcs"])
print("Confidence:", result["confidence"])
print("Threshold probabilities:", result["threshold_probabilities"])
```

For an already cropped anatomical-region image:

``` python
result = engine.predict_crop("path/to/cropped_image.jpg")

print("Predicted BCS:", result[0])
print("Confidence:", result[1])
print("Threshold probabilities:", result[2])
```

------------------------------------------------------------------------

## Running the GUI

The project includes a Tkinter graphical application:

``` text
inference/gui_app.py
```

From the project root, run:

``` bash
python -m inference.gui_app
```

The GUI allows the user to:

1.  Select a cow image
2.  Run the YOLO anatomical-region detector
3.  View the detected region
4.  Run the BCS prediction
5.  View the predicted BCS
6.  View prediction certainty
7.  Inspect ordinal threshold probabilities
8.  Optionally display the actual BCS when the selected image belongs to
    one of the recognized BCS directory structures

The GUI requires the trained detector and ordinal classifier checkpoints
to exist at the expected paths.

------------------------------------------------------------------------

## Environment Verification

Before training or inference, run:

``` bash
python test_env.py
```

Example information reported by this utility:

``` text
OS
Python version
Python interpreter path
PyTorch version
CPU/GPU availability
```

This is particularly useful when diagnosing CUDA, PyTorch, or
virtual-environment issues.

------------------------------------------------------------------------

## Reproducibility Notes

For a reproducible setup:

-   Use the same dataset version.
-   Keep the BCS class definitions unchanged unless the label mapping is
    intentionally modified.
-   Preserve the generated train/validation/test split CSV files when
    comparing experiments.
-   Keep the random seed used by `data/split.py`.
-   Record the exact Python and package versions used for training.
-   Keep the trained detector checkpoint and classifier checkpoint
    associated with the reported evaluation results.
-   Do not mix a detector checkpoint with a classifier checkpoint
    trained under incompatible preprocessing or dataset versions.

------------------------------------------------------------------------

## Important Implementation Details

### BCS mapping

The project currently maps:

``` python
{
    3.25: 0,
    3.50: 1,
    3.75: 2,
    4.00: 3,
    4.25: 4
}
```

### Input resolution

Classification models receive:

``` text
224 × 224
```

after aspect-ratio-preserving preprocessing.

### Ordinal thresholds

For five BCS classes, the ordinal model uses four binary thresholds.

The inference engine counts thresholds whose sigmoid probability exceeds
`0.5` and maps the resulting class index back to the corresponding BCS
value.

### Detector resolution

YOLO training/evaluation currently uses:

``` text
640 × 640
```

### Detector confidence threshold during inference

The two-stage inference engine uses:

``` text
conf_threshold = 0.35
```

unless another value is explicitly supplied.

------------------------------------------------------------------------

## Troubleshooting

### `FileNotFoundError` for the classifier checkpoint

Make sure the following file exists:

``` text
ReNet18_ordinal_classification_model/model_results_round2/best_ordinal_resnet18.pth
```

If it does not exist, run the ordinal training/search stage first.

### `YOLO detector is not initialized`

Make sure:

``` text
anatomical_region_ditector/runs/detect/anatomical_region_yolo11m/weights/best.pt
```

exists.

If not, prepare the detector dataset and train YOLO11m.

### Dataset images cannot be found

Check that:

``` text
dataset/
```

and/or:

``` text
cropped_dataset/
```

are located at the project root as expected.

Also verify the paths stored in:

``` text
data/meta_data/all_images.csv
data/meta_data/splits/train.csv
data/meta_data/splits/val.csv
data/meta_data/splits/test.csv
```

### CUDA is unavailable

The project can fall back to CPU through PyTorch/Ultralytics. CPU
training, especially for YOLO11m and repeated hyperparameter searches,
can be substantially slower.

### GUI does not start

Check:

-   Python installation
-   Tkinter availability
-   virtual environment activation
-   installed Pillow/PyTorch dependencies
-   existence of the required model checkpoints


------------------------------------------------------------------------

## Team Members

- **Fatemeh Sadat Ghazanfari** 
  - https://github.com/Fatemeh-S-Gh

- **Fatemeh Fahimpour** 

  - https://github.com/fatemefahimpour               
  

------------------------------------------------------------------------

## Suggested Future Improvements

### 1. Model Architecture Benchmarking
- **Title:** Investigation of Alternative Architectures
- **Current limitation:** Reliance on a single architecture (ResNet18) limits the potential to capture more complex patterns in anatomical features.
- **Proposed solution:** Experiment with state-of-the-art architectures, such as Vision Transformers (ViTs), EfficientNet, or ConvNeXt, to benchmark performance against the current baseline.
- **Expected benefit:** Potentially higher accuracy, improved feature extraction, and more robust classification for borderline BCS cases.


### 2. Domain-Specific Transfer Learning
- **Title:** Fine-tuning on Farm-Specific Data
- **Current limitation:** The base model may lack specific adaptation to unique lighting, environmental conditions, or breed characteristics of a target farm.
- **Proposed solution:** Implement Transfer Learning by freezing the pre-trained backbone of the best-performing model and fine-tuning the final classification head (or last few layers) on a curated, small-scale dataset obtained directly from the target deployment site.
- **Expected benefit:** Significantly improved inference precision tailored to the specific operational environment of the farm.

### 3. Cross-Dataset Generalization & Label Scope Expansion
- **Title:** Incorporation of Diverse Datasets and Full-Scale BCS Benchmarking
- **Current limitation:** The existing dataset exhibits significant structural constraints:
  - **Restricted Score Range:** Covers a narrow BCS window (3.25 to 4.25), whereas the standard BCS scale spans from 1.0 to 5.0, limiting the model's utility for severely under-conditioned or over-conditioned cattle.
  - **Label Noise & Inter-Observer Bias:** Potential inaccuracies or subjectivity in ground-truth annotations from the original dataset.
  - **Demographic Discrepancy:** The specific cattle breeds and morphological profiles in the current dataset may not adequately represent target commercial dairy farms.
- **Proposed solution:** Acquire, cross-validate, and integrate diverse public and partner agricultural datasets spanning the entire 1.0–5.0 BCS range. Re-train and benchmark baseline classifiers on multi-source datasets reflecting broader breed variations and robust consensus-based annotations.
- **Expected benefit:** Enables end-to-end full-scale BCS assessment, mitigates annotation noise, and significantly improves cross-farm generalizability across varying livestock breeds.


### 4. Anatomical Crop Margin Optimization
- **Title:** Optimization of Crop Padding and Contextual Noise Reduction
- **Current limitation:** The current cropping strategy utilizes an enlarged bounding box around the detected anatomical region to ensure context is captured. This potentially introduces excessive background noise or extraneous environmental features, which may dilute the signal of critical BCS-related markers for the classifier.
- **Proposed solution:** Conduct a systematic study on bounding box padding (margin) sizes. Benchmark classification performance across a range of crop configurations—from tight anatomical cuts to broader context-inclusive crops—to determine the optimal balance between sufficient context inclusion and feature clarity.
- **Expected benefit:** Improved Signal-to-Noise Ratio (SNR) for the classification model, reduced background interference, and sharper focus on high-fidelity anatomical morphology (e.g., bone structure contours).




------------------------------------------------------------------------



## Acknowledgements

This project was conducted as part of an internship program at the **Cyber-Physical Systems (CPS) Research Cluster**, **University of Isfahan**.

We would like to express our gratitude to:
- **Dr. Mahdi Kalbasi**: For his invaluable mentorship and guidance throughout this project, and for providing the research environment and necessary computational resources to conduct this study.
- **CPS Research Cluster, University of Isfahan**: For granting us the opportunity to undertake our internship within this stimulating research environment and for fostering our professional and technical growth.
- **Dataset Source**: We acknowledge the providers of the open-access cattle dataset hosted on [ScienceDB](https://www.scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c), which served as the foundational data for training and evaluating our pipeline.



------------------------------------------------------------------------

## Citation

If you find this project useful for your research or development, please consider citing it using the following format:
```text
"AI-Based Cow Body Condition Score (BCS) Estimation." 
Cyber-Physical Systems (CPS) Research Cluster, University of Isfahan, 2026.

