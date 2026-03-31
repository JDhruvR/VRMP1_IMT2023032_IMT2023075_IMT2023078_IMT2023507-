# Multi-Object Apparel Detection & Instance Segmentation

## Team Members
* **Dhruv Ramesh Joshi** – IMT2023032
* **Ankith Kini** – IMT2023075
* **Arnav Oruganty** – IMT2023078
* **Mithilesh Sai Yechuri** – IMT2023507

This repository contains the training, evaluation, and inference code for a visual recognition system designed to classify, detect, and segment clothing items in images. It is built on a subset of the **DeepFashion2** dataset, focusing on the 5 most frequent categories: *short sleeve top, long sleeve top, trousers, shorts, and skirt*.

## Project Tasks
1. **Image-Level Classification (Task 3.1):** Multi-label classification to identify all clothing categories present in an image.
2. **Detection & Segmentation (Task 3.2):** Predicting bounding boxes and instance-level pixel masks for each clothing item.

## Repository Structure

```text
.
├── efficientnet/      # Classification training (Scratch & Fine-tuned)
├── mobilenet/         # Classification training (Scratch & Fine-tuned)
├── resnet/            # Classification training (ResNet-50 Scratch & Fine-tuned)
├── mask-rcnn/         # Det/Seg training (Mask R-CNN Scratch & Fine-tuned)
├── unet/              # Segmentation training (U-Net with connected components)
├── yolo/              # Det/Seg training (YOLOv8-seg Scratch & Fine-tuned)
└── Inference_MP1/     # Final inference pipeline and validation scripts
    ├── run_inference.py 
    └── VRMP1/         # Payload containing predictor.py and final model weights
```

## Architectures Explored
For both tasks, models were trained from scratch and fine-tuned from ImageNet/COCO weights to establish baselines and compare convergence.
*   **Classification:** ResNet-50 (Selected Best), EfficientNet-B0/B2, MobileNetV3.
*   **Detection & Segmentation:** YOLOv8-seg (Selected Best), Mask R-CNN, U-Net (with post-processing for instance extraction).

## Evaluation Metrics
*   **Classification:** Macro/Micro F1-score, Precision, Recall, AUC-ROC.
*   **Detection:** COCO-style mAP@[0.5:0.95].
*   **Segmentation:** Per-class mIoU (macro-averaged) and Dice Coefficient.

## Running Inference Locally
The `Inference_MP1/VRMP1` directory contains the production-ready inference payload. 

1. Ensure the dataset is structured correctly inside `Inference_MP1/hidden_dataset/`.
2. Install requirements:
   ```bash
   pip install -r Inference_MP1/VRMP1/requirements.txt
   ```
3. Run the local validator to verify the pipeline and output formats:
   ```bash
   python Inference_MP1/VRMP1/validator_local.py
   ```

*Note: The best-performing model weights have been uploaded to Hugging Face due to GitHub file size limits.*
