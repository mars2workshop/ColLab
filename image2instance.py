"""
Instance Detection from Images using YOLOv5

This module provides functionality to detect object instances in a collection of images
using the YOLOv5 object detection model. The detected instances are filtered by confidence
threshold and saved to a CSV file for further analysis.

"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

import torch
import pandas as pd


class InstanceDetector:
    """
    A class for detecting object instances in images using YOLOv5 model.
    
    This class encapsulates the functionality to load a pre-trained YOLOv5 model,
    process multiple images, and extract object detection results with confidence filtering.
    """
    
    def __init__(self, model_name: str = "yolov5s", confidence_threshold: float = 0.5):
        """
        Initialize the InstanceDetector.
        
        Args:
            model_name (str): YOLOv5 model variant to use. Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
            confidence_threshold (float): Minimum confidence score for detections to be retained
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.all_detections = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> None:
        """
        Load the YOLOv5 model from torch hub.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self.logger.info(f"Loading YOLOv5 model: {self.model_name}")
            self.model = torch.hub.load("ultralytics/yolov5", self.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def detect_instances_in_image(self, image_path: str) -> pd.DataFrame:
        """
        Detect instances in a single image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            pd.DataFrame: Filtered detection results with confidence > threshold
            
        Raises:
            ValueError: If image cannot be processed
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            # Run inference on the image
            results = self.model(image_path)
            
            # Extract detection results as DataFrame
            detections_df = results.pandas().xyxy[0]
            
            # Add filename column for tracking
            filename = os.path.basename(image_path)
            detections_df['filename'] = filename
            
            # Filter detections by confidence threshold
            filtered_detections = detections_df[detections_df['confidence'] > self.confidence_threshold]
            
            self.logger.info(f"Processed {filename}: {len(filtered_detections)} detections above threshold")
            
            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            raise ValueError(f"Failed to process image {image_path}: {e}")
    
    def process_image_directory(self, images_dir: str) -> pd.DataFrame:
        """
        Process all images in a directory and detect instances.
        
        Args:
            images_dir (str): Path to directory containing images
            
        Returns:
            pd.DataFrame: Combined detection results from all images
            
        Raises:
            FileNotFoundError: If images directory doesn't exist
            ValueError: If no valid images found in directory
        """
        images_path = Path(images_dir)
        
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Get list of image files (common image extensions)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No valid image files found in {images_dir}")
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in image_files:
            try:
                detections = self.detect_instances_in_image(str(image_file))
                if not detections.empty:
                    self.all_detections.append(detections)
            except Exception as e:
                self.logger.warning(f"Skipping {image_file.name}: {e}")
                continue
        
        if not self.all_detections:
            raise ValueError("No detections found in any image")
        
        # Combine all detection results
        combined_detections = pd.concat(self.all_detections, ignore_index=True)
        
        self.logger.info(f"Total detections: {len(combined_detections)}")
        
        return combined_detections
    
    def save_detections(self, detections_df: pd.DataFrame, output_file: str) -> None:
        """
        Save detection results to CSV file.
        
        Args:
            detections_df (pd.DataFrame): Detection results to save
            output_file (str): Output CSV file path
        """
        try:
            detections_df.to_csv(output_file, index=False)
            self.logger.info(f"Detection results saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


def main():
    """
    Main function to execute the instance detection pipeline.
    """
    # Configuration parameters
    IMAGES_DIRECTORY = "images"
    OUTPUT_FILE = "detections.csv"
    MODEL_NAME = "yolov5s"
    CONFIDENCE_THRESHOLD = 0.5
    
    try:
        # Initialize detector
        detector = InstanceDetector(
            model_name=MODEL_NAME,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # Load the model
        detector.load_model()
        
        # Process all images in the directory
        all_detections = detector.process_image_directory(IMAGES_DIRECTORY)
        
        # Save results
        detector.save_detections(all_detections, OUTPUT_FILE)
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Total detections: {len(all_detections)}")
        print(f"📁 Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
