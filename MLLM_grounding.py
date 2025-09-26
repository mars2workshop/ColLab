"""
Multi-Modal Large Language Model (MLLM) Grounding Module

This module provides functionality to process cropped object instances through various
Multi-modal Large Language Models (MLLMs) to generate textual descriptions. The module
supports multiple models including Fuyu-8B, Qwen-VL variants, and provides concurrent
processing capabilities for efficient batch processing.

The module extracts visual characteristics from object instances and generates
natural language descriptions using state-of-the-art vision-language models.

"""

import base64
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass

import requests
from openai import OpenAI

from LMM.api_fuyu8B import get_access_token


@dataclass
class ModelResult:
    """
    Data class to store model processing results.
    
    Attributes:
        model_name (str): Name of the MLLM model used
        description (str): Generated description text
        processing_time (float): Time taken for processing in seconds
        file_path (str): Path to the processed image file
        success (bool): Whether processing was successful
        error_message (Optional[str]): Error message if processing failed
    """
    model_name: str
    description: str
    processing_time: float
    file_path: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class MLLMConfig:
    """
    Configuration class for MLLM models.
    
    Attributes:
        api_key (str): API key for the model
        base_url (str): Base URL for the API endpoint
        model_name (str): Specific model identifier
        max_retries (int): Maximum number of retry attempts
        timeout (int): Request timeout in seconds
    """
    api_key: str
    base_url: str
    model_name: str
    max_retries: int = 3
    timeout: int = 30


class ImageEncoder:
    """
    Utility class for image encoding operations.
    """
    
    @staticmethod
    def encode_to_base64(file_path: str, mime_type: str = "image/jpeg") -> str:
        """
        Convert image file to base64 encoded string.
        
        Args:
            file_path (str): Path to the image file
            mime_type (str): MIME type of the image
            
        Returns:
            str: Base64 encoded image with data URI prefix
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If image file cannot be read
        """
        try:
            with open(file_path, "rb") as image_file:
                encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_str}"
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {file_path}")
        except Exception as e:
            raise IOError(f"Failed to read image file {file_path}: {e}")


class MLLMProcessor:
    """
    Main class for processing images through Multiple Multi-modal Large Language Models.
    
    This class handles the orchestration of various MLLM models, manages concurrent
    processing, and provides comprehensive error handling and logging.
    """
    
    def __init__(self, crops_directory: str = "crops", output_file: str = "mllm_results.csv", 
                 max_workers: int = 8):
        """
        Initialize the MLLM processor.
        
        Args:
            crops_directory (str): Directory containing cropped images
            output_file (str): Output CSV file for results
            max_workers (int): Maximum number of concurrent threads
        """
        self.crops_directory = Path(crops_directory)
        self.output_file = Path(output_file)
        self.max_workers = max_workers
        
        # Statistics tracking
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.processing_results = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize image encoder
        self.image_encoder = ImageEncoder()
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate input directories and configuration.
        
        Raises:
            FileNotFoundError: If crops directory doesn't exist
            ValueError: If no image files found
        """
        if not self.crops_directory.exists():
            raise FileNotFoundError(f"Crops directory not found: {self.crops_directory}")
        
        # Count total images
        self.total_images = sum(1 for subdir in self.crops_directory.iterdir() 
                               if subdir.is_dir() 
                               for img_file in subdir.iterdir() 
                               if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
        
        if self.total_images == 0:
            raise ValueError(f"No image files found in {self.crops_directory}")
        
        self.logger.info(f"Found {self.total_images} images to process")
    
    def _extract_class_name(self, image_filename: str) -> str:
        """
        Extract class name from image filename.
        
        Args:
            image_filename (str): Name of the image file
            
        Returns:
            str: Extracted class name
        """
        return image_filename.split("_")[0]
    
    def _generate_prompt(self, class_name: str) -> str:
        """
        Generate prompt for MLLM based on class name.
        
        Args:
            class_name (str): Object class name
            
        Returns:
            str: Generated prompt text
        """
        return f"What are the characteristics of the {class_name} in this image? Just answer in one sentence."
    
    def process_with_fuyu8b(self, image_path: str, image_name: str) -> ModelResult:
        """
        Process image using Fuyu-8B model.
        
        Args:
            image_path (str): Path to the image file
            image_name (str): Name of the image file
            
        Returns:
            ModelResult: Processing result with description and metadata
        """
        start_time = time.time()
        
        try:
            # Extract class name and generate prompt
            class_name = self._extract_class_name(image_name)
            prompt = self._generate_prompt(class_name)
            
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            encoded_string = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare API request
            url = ("https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/image2text/fuyu_8b"
                   f"?access_token={get_access_token()}")
            
            payload = json.dumps({
                "prompt": prompt,
                "image": encoded_string
            })
            
            headers = {'Content-Type': 'application/json'}
            
            # Make API request
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            description = response_data.get("result", "No description available")
            
            processing_time = time.time() - start_time
            
            return ModelResult(
                model_name="fuyu8b",
                description=description.rstrip('\n'),
                processing_time=round(processing_time, 4),
                file_path=image_path,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Fuyu-8B processing failed for {image_name}: {e}")
            
            return ModelResult(
                model_name="fuyu8b",
                description="Processing failed",
                processing_time=round(processing_time, 4),
                file_path=image_path,
                success=False,
                error_message=str(e)
            )
    
    def _process_with_qwen_model(self, image_path: str, image_name: str, 
                                model_name: str, config: MLLMConfig) -> ModelResult:
        """
        Generic method to process image using Qwen-VL models.
        
        Args:
            image_path (str): Path to the image file
            image_name (str): Name of the image file
            model_name (str): Model identifier
            config (MLLMConfig): Model configuration
            
        Returns:
            ModelResult: Processing result with description and metadata
        """
        start_time = time.time()
        
        try:
            # Extract class name and generate prompt
            class_name = self._extract_class_name(image_name)
            prompt = self._generate_prompt(class_name)
            
            # Prepare messages for OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.image_encoder.encode_to_base64(image_path)
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            # Initialize OpenAI client
            client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
            
            # Make API request
            completion = client.chat.completions.create(
                model=config.model_name,
                messages=messages
            )
            
            description = completion.choices[0].message.content
            processing_time = time.time() - start_time
            
            return ModelResult(
                model_name=model_name,
                description=description.rstrip('\n') if description else "No description available",
                processing_time=round(processing_time, 4),
                file_path=image_path,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"{model_name} processing failed for {image_name}: {e}")
            
            return ModelResult(
                model_name=model_name,
                description="Processing failed",
                processing_time=round(processing_time, 4),
                file_path=image_path,
                success=False,
                error_message=str(e)
            )
    
    def process_with_qwen_3b(self, image_path: str, image_name: str) -> ModelResult:
        """Process image using Qwen2.5-VL-3B model."""
        config = MLLMConfig(
            api_key='',  # Add your API key here
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen2.5-vl-3b-instruct"
        )
        return self._process_with_qwen_model(image_path, image_name, "qwen3b", config)
    
    def process_with_qwen_7b(self, image_path: str, image_name: str) -> ModelResult:
        """Process image using Qwen2.5-VL-7B model."""
        config = MLLMConfig(
            api_key='',  # Add your API key here
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen2.5-vl-7b-instruct"
        )
        return self._process_with_qwen_model(image_path, image_name, "qwen7b", config)
    
    def process_with_qwen_32b(self, image_path: str, image_name: str) -> ModelResult:
        """Process image using Qwen2.5-VL-32B model."""
        config = MLLMConfig(
            api_key='',  # Add your API key here
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen2.5-vl-32b-instruct"
        )
        return self._process_with_qwen_model(image_path, image_name, "qwen32b", config)
    
    def process_single_image(self, subfolder: str, image_name: str) -> List[ModelResult]:
        """
        Process a single image through multiple MLLM models.
        
        Args:
            subfolder (str): Subfolder name containing the image
            image_name (str): Name of the image file
            
        Returns:
            List[ModelResult]: Results from all enabled models
        """
        image_path = str(self.crops_directory / subfolder / image_name)
        relative_path = str(Path(subfolder) / image_name)
        
        results = []
        
        # Check if file exists
        if not Path(image_path).is_file():
            self.logger.warning(f"Image file not found: {image_path}")
            return results
        
        # Process with enabled models (configure which models to use)
        enabled_models = [
            self.process_with_fuyu8b,
            # self.process_with_qwen_3b,
            # self.process_with_qwen_7b,
            # self.process_with_qwen_32b,
        ]
        
        for model_func in enabled_models:
            try:
                result = model_func(image_path, image_name)
                # Update file path to relative path for CSV output
                result.file_path = relative_path
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_name} with {model_func.__name__}: {e}")
        
        return results
    
    def collect_image_paths(self) -> List[Tuple[str, str]]:
        """
        Collect all image file paths for processing.
        
        Returns:
            List[Tuple[str, str]]: List of (subfolder, image_name) tuples
        """
        image_infos = []
        
        for subfolder in sorted(self.crops_directory.iterdir()):
            if not subfolder.is_dir():
                continue
                
            for image_file in sorted(subfolder.iterdir()):
                if (image_file.is_file() and 
                    image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
                    image_infos.append((subfolder.name, image_file.name))
        
        return image_infos
    
    def process_all_images(self) -> Dict[str, Any]:
        """
        Process all images through MLLMs using concurrent processing.
        
        Returns:
            Dict[str, Any]: Processing statistics and results
        """
        # Collect all image paths
        image_infos = self.collect_image_paths()
        self.logger.info(f"Processing {len(image_infos)} images with {self.max_workers} workers")
        
        # Process images concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_img = {
                executor.submit(self.process_single_image, subfolder, img_name): (subfolder, img_name)
                for subfolder, img_name in image_infos
            }
            
            # Collect results
            for future in as_completed(future_to_img):
                subfolder, img_name = future_to_img[future]
                try:
                    results = future.result()
                    self.processing_results.extend(results)
                    self.processed_images += 1
                    
                    # Log progress
                    if self.processed_images % 10 == 0 or self.processed_images == len(image_infos):
                        self.logger.info(f"Progress: {self.processed_images}/{len(image_infos)} images processed")
                    
                except Exception as exc:
                    self.failed_images += 1
                    self.logger.error(f"Processing failed for {subfolder}/{img_name}: {exc}")
        
        # Calculate statistics
        successful_results = [r for r in self.processing_results if r.success]
        failed_results = [r for r in self.processing_results if not r.success]
        
        statistics = {
            'total_images': len(image_infos),
            'processed_images': self.processed_images,
            'failed_images': self.failed_images,
            'total_model_calls': len(self.processing_results),
            'successful_model_calls': len(successful_results),
            'failed_model_calls': len(failed_results),
            'success_rate': len(successful_results) / max(1, len(self.processing_results)) * 100,
            'average_processing_time': sum(r.processing_time for r in successful_results) / max(1, len(successful_results))
        }
        
        return statistics
    
    def save_results_to_csv(self) -> None:
        """
        Save processing results to CSV file.
        """
        try:
            with open(self.output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["model", "description", "processing_time", "file_path", "success", "error_message"])
                
                for result in self.processing_results:
                    writer.writerow([
                        result.model_name,
                        result.description,
                        result.processing_time,
                        result.file_path,
                        result.success,
                        result.error_message or ""
                    ])
            
            self.logger.info(f"Results saved to: {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def generate_summary_report(self, statistics: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            statistics (Dict[str, Any]): Processing statistics
            
        Returns:
            str: Formatted summary report
        """
        report = f"""
        MLLM Processing Summary Report
        =============================

        Configuration:
        - Crops Directory: {self.crops_directory}
        - Output File: {self.output_file}
        - Max Workers: {self.max_workers}

        Processing Results:
        - Total Images: {statistics['total_images']}
        - Processed Images: {statistics['processed_images']}
        - Failed Images: {statistics['failed_images']}
        - Total Model Calls: {statistics['total_model_calls']}
        - Successful Model Calls: {statistics['successful_model_calls']}
        - Failed Model Calls: {statistics['failed_model_calls']}

        Performance Metrics:
        - Success Rate: {statistics['success_rate']:.1f}%
        - Average Processing Time: {statistics['average_processing_time']:.3f} seconds
        - Total Processing Time: {sum(r.processing_time for r in self.processing_results):.1f} seconds

        Model Performance:
        """
        
        # Add model-specific statistics
        model_stats = {}
        for result in self.processing_results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {'successful': 0, 'failed': 0, 'total_time': 0}
            
            if result.success:
                model_stats[result.model_name]['successful'] += 1
                model_stats[result.model_name]['total_time'] += result.processing_time
            else:
                model_stats[result.model_name]['failed'] += 1
        
        for model_name, stats in model_stats.items():
            total_calls = stats['successful'] + stats['failed']
            success_rate = (stats['successful'] / max(1, total_calls)) * 100
            avg_time = stats['total_time'] / max(1, stats['successful'])
            
            report += f"- {model_name}: {stats['successful']}/{total_calls} successful ({success_rate:.1f}%), "
            report += f"avg time: {avg_time:.3f}s\n"
        
        return report


def main():
    """
    Main function to execute the MLLM processing pipeline.
    """
    # Configuration parameters
    CROPS_DIRECTORY = "crops"
    OUTPUT_FILE = "mllm_grounding_results.csv"
    MAX_WORKERS = 8
    
    try:
        print("🔄 Starting MLLM processing pipeline...")
        
        # Initialize processor
        processor = MLLMProcessor(
            crops_directory=CROPS_DIRECTORY,
            output_file=OUTPUT_FILE,
            max_workers=MAX_WORKERS
        )
        
        # Process all images
        statistics = processor.process_all_images()
        
        # Save results
        processor.save_results_to_csv()
        
        # Generate and display summary report
        report = processor.generate_summary_report(statistics)
        print(report)
        
        print("✅ MLLM processing completed successfully!")
        print(f"📊 Total model calls: {statistics['total_model_calls']}")
        print(f"📁 Results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
