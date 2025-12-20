from glob import glob
import logging
import ray
import torch
from PIL import Image
from typing import List, Tuple, Union, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class DinoV2ActorBatched:
    def __init__(self, model_name="facebook/dinov2-base", use_gpu=False, batch_size=16, auto_batch_size=False):
        """
        Initialize the DinoV2 model with batching capabilities.
        
        Args:
            model_name: The name or path of the DinoV2 model to use
            use_gpu: Whether to use GPU for inference
            batch_size: Maximum number of images to process in a single batch
            auto_batch_size: If True, automatically determine the maximum batch size based on GPU memory
        """
        from transformers import AutoImageProcessor, AutoModel

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Determine batch size
        if auto_batch_size and self.device == "cuda":
            self.batch_size = self._determine_max_batch_size()
            logging.info(f"Auto-determined maximum batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
            
        # Ensure return_dict is False for compatibility with torch.jit.trace
        self.model.config.return_dict = False
        
        logging.info(f"DinoV2ActorBatched initialized with device={self.device}, batch_size={self.batch_size}")

    def _determine_max_batch_size(self, start_batch_size=4, max_batch_size=128, safety_factor=0.9):
        """
        Automatically determine the maximum batch size that can fit in GPU memory.
        
        Args:
            start_batch_size: Initial batch size to try
            max_batch_size: Maximum batch size to consider
            safety_factor: Factor to reduce the final batch size by for safety margin
            
        Returns:
            Maximum batch size that can fit in GPU memory
        """
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        # Start with a small batch size and increase until we get OOM
        current_batch_size = start_batch_size
        max_successful_batch = start_batch_size
        
        while current_batch_size <= max_batch_size:
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Create a batch of dummy images
                dummy_batch = [dummy_image] * current_batch_size
                
                # Try processing the batch
                logging.info(f"Testing batch size: {current_batch_size}")
                inputs = self.processor(images=dummy_batch, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    self.model(**inputs)
                
                # If successful, update max successful batch and try a larger batch
                max_successful_batch = current_batch_size
                current_batch_size *= 2
                
            except RuntimeError as e:
                # If we get an OOM error, break the loop
                if "CUDA out of memory" in str(e):
                    logging.info(f"CUDA OOM at batch size {current_batch_size}")
                    break
                else:
                    # If it's another error, re-raise it
                    raise e
        
        # If we hit the max_batch_size without OOM, use that
        if max_successful_batch == max_batch_size:
            logging.info(f"Max batch size reached without OOM: {max_batch_size}")
            return max_batch_size
        
        # Binary search to find the exact maximum batch size
        low = max_successful_batch
        high = current_batch_size
        
        while low < high - 1:
            mid = (low + high) // 2
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Create a batch of dummy images
                dummy_batch = [dummy_image] * mid
                
                # Try processing the batch
                logging.info(f"Fine-tuning batch size: {mid}")
                inputs = self.processor(images=dummy_batch, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    self.model(**inputs)
                
                # If successful, update low
                low = mid
            except RuntimeError as e:
                # If we get an OOM error, update high
                if "CUDA out of memory" in str(e):
                    high = mid
                else:
                    # If it's another error, re-raise it
                    raise e
        
        # Apply safety factor and return as integer
        logging.info(f"Final batch size: {low}")
        return max(1, int(low * safety_factor))

    def _process_batch(self, image_batch: List[Image.Image]):
        """Process a batch of images and return their embeddings."""
        # Process images as a batch
        inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs[0]

        return outputs, last_hidden_states

    def predict_batch(self, images: List[Image.Image]):
        """
        Process multiple images in batches and return their embeddings.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            Tuple of (outputs, last_hidden_states) for all images
        """
        if not images:
            return None, None
            
        all_outputs = []
        all_hidden_states = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            logging.info(f"Processing batch {i//self.batch_size + 1}/{(len(images) + self.batch_size - 1)//self.batch_size} with {len(batch)} images")
            
            outputs, hidden_states = self._process_batch(batch)
            
            all_outputs.append(outputs)
            all_hidden_states.append(hidden_states)
        
        # Combine results from all batches
        # Note: The exact way to combine depends on how you want to use the outputs
        combined_hidden_states = torch.cat(all_hidden_states, dim=0)
        
        return all_outputs, combined_hidden_states
    
    def predict(self, image_pil: Image.Image):
        """
        Process a single image for backward compatibility with the original DinoV2Actor.
        
        Args:
            image_pil: A single PIL Image to process
            
        Returns:
            Tuple of (outputs, last_hidden_states) for the image
        """
        outputs, hidden_states = self._process_batch([image_pil])
        return outputs, hidden_states[0]  # Return the first (and only) item


if __name__ == "__main__":
    import requests
    import matplotlib.pyplot as plt
    
    #ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus
    server_address = "ray://grail-mercury.neu.edu:10001"
    ignore = glob("*/")
    ignore = [item for item in ignore if "src" not in item]
    ignore = ignore + glob("src/*/")
    ignore = [item for item in ignore if "models" not in item]
    ignore = ["\\" + item for item in ignore]
    ignore.append('\\.git\\')

    ray.init(address=server_address, runtime_env={"working_dir": ".", "excludes": ignore})

    use_gpu = True

    # URL of the image to process
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # Load image from URL
    image = requests.get(url, stream=True).raw
    # Load image with PIL
    image_pil = Image.open(image)
    
    # Create a batch of the same image for demonstration
    batch_size = 10
    image_batch = [image_pil] * batch_size

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance with dynamic GPU allocation and auto batch size
    dinov2_actor = DinoV2ActorBatched.options(**actor_options).remote(
        use_gpu=use_gpu, 
        auto_batch_size=True  # Enable automatic batch size determination
    )

    # Test single image prediction (backward compatibility)
    print("\n--- Single Image Processing ---")
    single_start_time = time.time()
    single_future = dinov2_actor.predict.remote(image_pil)
    outputs, features = ray.get(single_future)
    single_end_time = time.time()
    single_time = single_end_time - single_start_time
    print(f"Single image features shape: {features.shape}")
    print(f"Single image processing time: {single_time:.4f} seconds")
    
    # Test sequential processing of multiple images (one by one)
    print("\n--- Sequential Processing of Multiple Images ---")
    seq_start_time = time.time()
    for img in image_batch:
        _ = ray.get(dinov2_actor.predict.remote(img))
    seq_end_time = time.time()
    seq_time = seq_end_time - seq_start_time
    print(f"Sequential processing of {batch_size} images: {seq_time:.4f} seconds")
    print(f"Average time per image: {seq_time/batch_size:.4f} seconds")

    # Test batch prediction
    print("\n--- Batch Processing of Multiple Images ---")
    batch_start_time = time.time()
    batch_future = dinov2_actor.predict_batch.remote(image_batch)
    batch_outputs, batch_features = ray.get(batch_future)
    batch_end_time = time.time()
    batch_time = batch_end_time - batch_start_time
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Number of images in batch: {batch_features.shape[0]}")
    print(f"Batch processing time for {batch_size} images: {batch_time:.4f} seconds")
    print(f"Average time per image in batch: {batch_time/batch_size:.4f} seconds")
    
    # Calculate and display speedup
    speedup = seq_time / batch_time
    print(f"\n--- Performance Comparison ---")
    print(f"Speedup from batch processing: {speedup:.2f}x faster than sequential processing")
    
    # Plot
    try:
        labels = ['Single Image', 'Sequential (per image)', 'Batch (per image)']
        times = [single_time, seq_time/batch_size, batch_time/batch_size]
        
        plt.figure(figsize=(10, 5))
        plt.bar(labels, times)
        plt.ylabel('Time (seconds)')
        plt.title('DinoV2 Processing Time Comparison')
        plt.show()
        print(f"Performance comparison plot saved to 'dinov2_performance_comparison.png'")
    except Exception as e:
        print(f"Could not create performance plot: {e}")

    # Shutdown Ray
    ray.shutdown() 