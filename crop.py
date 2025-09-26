"""
Image Instance Cropping Module

This module provides functionality to extract and crop object instances from images
based on detection results. The detected bounding boxes are used to crop individual
object instances and save them as separate image files for further analysis.

The module processes detection results from CSV files containing bounding box coordinates,
crops the corresponding regions from original images, and organizes the cropped instances
by image and class type.

"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

import cv2
import pandas as pd
import numpy as np


class InstanceCropper:
    """
    A class for cropping object instances from images based on detection results.
    
    This class processes detection results stored in CSV format and extracts
    individual object instances by cropping the corresponding bounding box regions
    from the original images.
    """
    
    def __init__(self, csv_path: str, image_dir: str, output_dir: str = "crops"):
        """
        Initialize the InstanceCropper.
        
        Args:
            csv_path (str): Path to CSV file containing detection results
            image_dir (str): Directory containing the original images
            output_dir (str): Directory to save cropped instances
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        
        # Statistics tracking
        self.total_crops = 0
        self.processed_images = 0
        self.failed_images = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_inputs(self) -> None:
        """
        Validate input files and directories.
        
        Raises:
            FileNotFoundError: If required files or directories don't exist
            ValueError: If CSV file is empty or has invalid format
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Detection CSV file not found: {self.csv_path}")
            
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        
        # Validate CSV format
        try:
            df = pd.read_csv(self.csv_path)
            required_columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")
                
            if df.empty:
                raise ValueError("Detection CSV file is empty")
                
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {e}")
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load an image from file path.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            Optional[np.ndarray]: Loaded image array or None if loading fails
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Failed to load image: {image_path.name}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path.name}: {e}")
            return None
    
    def _validate_bounding_box(self, bbox: Tuple[int, int, int, int], 
                              image_shape: Tuple[int, int, int]) -> bool:
        """
        Validate bounding box coordinates against image dimensions.
        
        Args:
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax)
            image_shape (Tuple[int, int, int]): Image shape (height, width, channels)
            
        Returns:
            bool: True if bounding box is valid, False otherwise
        """
        xmin, ymin, xmax, ymax = bbox
        height, width = image_shape[:2]
        
        # Check if coordinates are within image bounds
        if xmin < 0 or ymin < 0 or xmax >= width or ymax >= height:
            return False
            
        # Check if bounding box has positive area
        if xmax <= xmin or ymax <= ymin:
            return False
            
        return True
    
    def _crop_instance(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop an instance from the image using bounding box coordinates.
        
        Args:
            image (np.ndarray): Source image
            bbox (Tuple[int, int, int, int]): Bounding box coordinates (xmin, ymin, xmax, ymax)
            
        Returns:
            Optional[np.ndarray]: Cropped image region or None if cropping fails
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Validate bounding box
        if not self._validate_bounding_box(bbox, image.shape):
            self.logger.warning(f"Invalid bounding box: {bbox} for image shape {image.shape}")
            return None
        
        try:
            # Crop the image region
            cropped_image = image[ymin:ymax, xmin:xmax]
            
            # Verify crop is not empty
            if cropped_image.size == 0:
                self.logger.warning(f"Empty crop for bounding box: {bbox}")
                return None
                
            return cropped_image
            
        except Exception as e:
            self.logger.error(f"Error cropping region {bbox}: {e}")
            return None
    
    def _generate_crop_filename(self, class_name: str, instance_id: int, 
                               file_extension: str = ".jpg") -> str:
        """
        Generate filename for cropped instance.
        
        Args:
            class_name (str): Object class name
            instance_id (int): Instance identifier within the class
            file_extension (str): File extension for the cropped image
            
        Returns:
            str: Generated filename
        """
        # Sanitize class name for filename
        safe_class_name = "".join(c for c in class_name if c.isalnum() or c in ('-', '_'))
        return f"{safe_class_name}_{instance_id:03d}{file_extension}"
    
    def process_image_detections(self, filename: str, detections_group: pd.DataFrame) -> int:
        """
        Process all detections for a single image and crop instances.
        
        Args:
            filename (str): Name of the image file
            detections_group (pd.DataFrame): Detection results for this image
            
        Returns:
            int: Number of successfully cropped instances
        """
        image_path = self.image_dir / filename
        image_name_no_ext = image_path.stem
        
        # Create subdirectory for this image's crops
        image_output_dir = self.output_dir / image_name_no_ext
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the source image
        image = self._load_image(image_path)
        if image is None:
            self.failed_images.append(filename)
            return 0
        
        # Track class instances for numbering
        class_counters = defaultdict(int)
        crops_created = 0
        
        self.logger.info(f"Processing {filename}: {len(detections_group)} detections")
        
        # Process each detection
        for _, detection in detections_group.iterrows():
            try:
                # Extract bounding box coordinates
                xmin = int(detection['xmin'])
                ymin = int(detection['ymin'])
                xmax = int(detection['xmax'])
                ymax = int(detection['ymax'])
                class_name = str(detection['name'])
                
                bbox = (xmin, ymin, xmax, ymax)
                
                # Crop the instance
                cropped_instance = self._crop_instance(image, bbox)
                if cropped_instance is None:
                    continue
                
                # Update class counter and generate filename
                class_counters[class_name] += 1
                instance_id = class_counters[class_name]
                crop_filename = self._generate_crop_filename(class_name, instance_id)
                crop_path = image_output_dir / crop_filename
                
                # Save cropped instance
                success = cv2.imwrite(str(crop_path), cropped_instance)
                if success:
                    crops_created += 1
                    self.logger.debug(f"Saved crop: {crop_path}")
                else:
                    self.logger.warning(f"Failed to save crop: {crop_path}")
                    
            except Exception as e:
                self.logger.error(f"Error processing detection in {filename}: {e}")
                continue
        
        self.logger.info(f"Created {crops_created} crops for {filename}")
        return crops_created
    
    def crop_all_instances(self) -> Dict[str, int]:
        """
        Process all images and crop detected instances.
        
        Returns:
            Dict[str, int]: Statistics about the cropping process
        """
        try:
            # Load detection results
            self.logger.info(f"Loading detection results from: {self.csv_path}")
            detections_df = pd.read_csv(self.csv_path)
            
            total_detections = len(detections_df)
            unique_images = detections_df['filename'].nunique()
            
            self.logger.info(f"Processing {total_detections} detections from {unique_images} images")
            
            # Group detections by image filename
            grouped_detections = detections_df.groupby('filename')
            
            # Process each image
            for filename, group in grouped_detections:
                crops_created = self.process_image_detections(filename, group)
                self.total_crops += crops_created
                self.processed_images += 1
            
            # Calculate statistics
            statistics = {
                'total_images_processed': self.processed_images,
                'total_crops_created': self.total_crops,
                'failed_images': len(self.failed_images),
                'success_rate': (self.processed_images - len(self.failed_images)) / self.processed_images * 100
                if self.processed_images > 0 else 0
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error during cropping process: {e}")
            raise
    
    def generate_summary_report(self, statistics: Dict[str, int]) -> str:
        """
        Generate a summary report of the cropping process.
        
        Args:
            statistics (Dict[str, int]): Processing statistics
            
        Returns:
            str: Formatted summary report
        """
        report = f"""
        Instance Cropping Summary Report
        ================================

        Input Parameters:
        - Detection CSV: {self.csv_path}
        - Images Directory: {self.image_dir}
        - Output Directory: {self.output_dir}

        Processing Results:
        - Images Processed: {statistics['total_images_processed']}
        - Total Crops Created: {statistics['total_crops_created']}
        - Failed Images: {statistics['failed_images']}
        - Success Rate: {statistics['success_rate']:.1f}%

        Average Crops per Image: {statistics['total_crops_created'] / max(1, statistics['total_images_processed']):.1f}
        """
        
        if self.failed_images:
            report += f"\nFailed Images:\n"
            for failed_img in self.failed_images:
                report += f"- {failed_img}\n"
        
        return report


def main():
    """
    Main function to execute the instance cropping pipeline.
    """
    # Configuration parameters
    CSV_PATH = "detection.csv"
    IMAGE_DIRECTORY = "images"
    OUTPUT_DIRECTORY = "crops"
    
    try:
        print("🔄 Starting instance cropping pipeline...")
        
        # Initialize cropper
        cropper = InstanceCropper(
            csv_path=CSV_PATH,
            image_dir=IMAGE_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY
        )
        
        # Process all instances
        statistics = cropper.crop_all_instances()
        
        # Generate and display summary report
        report = cropper.generate_summary_report(statistics)
        print(report)
        
        print("✅ Instance cropping completed successfully!")
        print(f"📊 Total crops created: {statistics['total_crops_created']}")
        print(f"📁 Output directory: {OUTPUT_DIRECTORY}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
