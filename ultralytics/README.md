Citrus-YOLO11: A Lightweight and Robust Detector for Real-Time Citrus Detection

Introduction

In agricultural automation, identifying citrus fruits in complex orchard environments (such as severe occlusion and background interference) is crucial for robotic harvesting. This study proposes an enhanced Citrus-YOLO11 model to address the feature loss and false detection issues in previous YOLO series models.

Citrus-YOLO11 integrates several key innovations:
LLKFE: Enlarges the receptive field to handle occluded targets.
AFR: Filters out complex orchard backgrounds.
DPSPP: Enhances multi-scale detection capabilities.
CAMA: Achieves efficient feature fusion and upsampling in the neck.

The Four Core Improvements

LLKFE (Lightweight Large-Kernel Feature Extraction)
We have detailed the LLKFE module code in ultralytics/nn/modules/block.py. This module implementation focuses on enlarging the receptive field to effectively handle targets under severe occlusion.

AFR (Adaptive Feature Refinement)
We have detailed the AFR module code in ultralytics/nn/modules/block.py. This module is specifically designed to filter out complex orchard backgrounds, such as leaves and branches, significantly reducing false positives.

DPSPP (Dual-Path Spatial Pyramid Pooling)
We have detailed the DPSPP module code in ultralytics/nn/modules/block.py. This structure is proposed to enhance the network's multi-scale feature extraction capabilities, ensuring robust detection across different fruit sizes.

CAMA (Content-Aware Multiscale Aggregation)
We have detailed the CAMA mechanism code in ultralytics/nn/modules/block.py. This module is crucial for the neck to achieve efficient feature fusion and upsampling, improving both accuracy and inference speed.

Network Architecture
The complete network architecture is defined in ultralytics/citrus_yolo11.yaml. We have added detailed comments to indicate the locations of these four modules and the overall data flow.

Get Started

Environment Setup
Set up the environment using the provided requirements file:
pip install -r requirements.txt

Dataset
Orange Dataset (Our Dataset): To be released / Link to be updated

Training & Testing

Train
To start training from scratch, configure your data path in your dataset configuration file (e.g., orange.yaml), then run:
python train.py

Test (FPS Evaluation)
The test.py script is used for performing inference and FPS evaluation on images using the trained model to verify real-time performance.
python test.py

