# ColLab

This project provides a complete pipeline for processing images through object detection, instance cropping, multi-modal description generation, and description augmentation using Large Language Models.

## Pipeline Overview

The pipeline consists of four main stages that should be executed in the following order:

### 1. Instance Detection (`image2instance.py`)
- **Purpose**: Extract object instances from images using YOLOv5 model
- **Input**: Directory of images (`images/`)
- **Output**: CSV file with detection results (`detections.csv`)
- **Description**: Processes all images in the input directory and detects object instances with confidence > 0.5

### 2. Instance Cropping (`crop.py`)
- **Purpose**: Crop detected object instances from original images
- **Input**: Detection CSV file and original images
- **Output**: Directory of cropped instances (`crops/`)
- **Description**: Creates individual image files for each detected instance, organized by source image

### 3. Multi-Modal Description Generation (`MLLM_grounding.py`)
- **Purpose**: Generate textual descriptions using Multiple Multi-modal Large Language Models
- **Input**: Cropped instance images
- **Output**: CSV file with model-generated descriptions (`mllm_grounding_results.csv`)
- **Description**: Processes cropped images through various MLLMs (Fuyu-8B, Qwen-VL variants) to generate descriptions

### 4. Description Augmentation (`LLM_augmentation.py`)
- **Purpose**: Enhance and consolidate descriptions using Large Language Models
- **Input**: MLLM-generated descriptions CSV
- **Output**: Augmented descriptions CSV (`description_augmentation_results.csv`)
- **Description**: Uses LLM to extract commonalities and generate improved descriptions for visual grounding tasks

## Quick Start

1. **Prepare your images**: Place all input images in the `images/` directory

2. **Run the pipeline in order**:
   ```bash
   # Step 1: Detect instances
   python image2instance.py
   
   # Step 2: Crop instances
   python crop.py
   
   # Step 3: Generate descriptions
   python MLLM_grounding.py
   
   # Step 4: Augment descriptions
   python LLM_augmentation.py
   ```

3. **Configure API keys**: Before running steps 3 and 4, make sure to add your API keys in the respective files

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Pandas
- OpenAI Python client
- Other dependencies as specified in each module

## Output Structure

```
├── images/                          # Input images
├── detections.csv         # Detection results
├── crops/                          # Cropped instances
│   ├── image1_name/
│   │   ├── class1_001.jpg
│   │   └── class2_001.jpg
│   └── image2_name/
├── mllm_grounding_results.csv      # MLLM descriptions
└── description_augmentation_results.csv  # Final augmented descriptions
```

## Notes

- Ensure sufficient disk space for cropped images
- API rate limits may affect processing speed in steps 3 and 4
- Each step generates detailed logs for monitoring progress
- The pipeline is designed for research and academic use