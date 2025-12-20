from glob import glob
import logging
import ray
import torch
from PIL import Image
from typing import List, Tuple, Union, Optional
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class Blipv2ActorBatched:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", use_gpu=False, batch_size=8, auto_batch_size=False):
        """
        Initialize the BLIP-2 model with batching capabilities.
        
        Args:
            model_name: The name or path of the BLIP-2 model to use
            use_gpu: Whether to use GPU for inference
            batch_size: Maximum number of images to process in a single batch
            auto_batch_size: If True, automatically determine the maximum batch size based on GPU memory
        """
        from transformers import Blip2Processor, Blip2Model, BitsAndBytesConfig

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Load the model with appropriate settings for the device
        if self.device == "cuda":
            # Use BitsAndBytesConfig instead of load_in_8bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            self.model = Blip2Model.from_pretrained(
                model_name, 
                device_map={"": 0}, 
                torch_dtype=torch.float16,
                quantization_config=quantization_config
            )
        else:
            self.model = Blip2Model.from_pretrained(model_name)
            self.model.to(self.device)
        
        # Determine batch size
        if auto_batch_size and self.device == "cuda":
            self.batch_size = self._determine_max_batch_size()
            logging.info(f"Auto-determined maximum batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
            
        logging.info(f"Blipv2ActorBatched initialized with device={self.device}, batch_size={self.batch_size}")

    def _determine_max_batch_size(self, start_batch_size=2, max_batch_size=32, safety_factor=0.9):
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
                    self.model.get_image_features(**inputs)
                
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
                    self.model.get_image_features(**inputs)
                
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

        # Perform inference to get image embeddings
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
            # Handle the output correctly based on its type
            if hasattr(outputs, "last_hidden_state"):
                # If it's a BaseModelOutputWithPooling object
                image_features = outputs.last_hidden_state
            else:
                # If it's a tensor
                image_features = outputs
            
            # Normalize the features
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy for easier handling
            embeddings_np = image_embeddings.cpu().numpy()

        return embeddings_np

    def predict_batch(self, images: List[Image.Image]):
        """
        Process multiple images in batches and return their embeddings.
        
        Args:
            images: List of PIL Image objects to process
            
        Returns:
            Numpy array of embeddings for all images
        """
        if not images:
            return None
            
        all_embeddings = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            logging.info(f"Processing batch {i//self.batch_size + 1}/{(len(images) + self.batch_size - 1)//self.batch_size} with {len(batch)} images")
            
            embeddings = self._process_batch(batch)
            all_embeddings.append(embeddings)
        
        # Combine results from all batches
        combined_embeddings = np.vstack(all_embeddings)
        
        return combined_embeddings
    
    def predict(self, image_pil: Image.Image):
        """
        Process a single image for backward compatibility.
        
        Args:
            image_pil: A single PIL Image to process
            
        Returns:
            Numpy array of embeddings for the image
        """
        embeddings = self._process_batch([image_pil])
        return embeddings[0]  # Return the first (and only) item


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import requests
    from io import BytesIO
    
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
    batch_size = 16
    
    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}
    
    # Create an instance of the Blipv2ActorBatched
    blipv2_actor = Blipv2ActorBatched.options(**actor_options).remote(
        use_gpu=use_gpu, 
        batch_size=batch_size,
        auto_batch_size=True
    )
    
    # Load sample images
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000000285.jpg",  # person on bench
        "http://images.cocodataset.org/val2017/000000578967.jpg",  # dog
        "http://images.cocodataset.org/val2017/000000093965.jpg",  # zebra
    ]
    
    image_batch = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image_batch.append(image)
    
    # Test single image prediction
    print("\n--- Single Image Processing ---")
    single_start_time = time.time()
    single_future = blipv2_actor.predict.remote(image_batch[0])
    single_embedding = ray.get(single_future)
    single_end_time = time.time()
    single_time = single_end_time - single_start_time
    print(f"Single image embedding shape: {single_embedding.shape}")
    print(f"Single image embedding dimensionality: {single_embedding.shape[-1]}")
    print(f"Single image processing time: {single_time:.4f} seconds")
    
    # Test sequential processing of multiple images
    print("\n--- Sequential Processing of Multiple Images ---")
    seq_start_time = time.time()
    seq_futures = [blipv2_actor.predict.remote(img) for img in image_batch]
    seq_embeddings = ray.get(seq_futures)
    seq_end_time = time.time()
    seq_time = seq_end_time - seq_start_time
    print(f"Sequential processing time for {len(image_batch)} images: {seq_time:.4f} seconds")
    print(f"Average time per image: {seq_time/len(image_batch):.4f} seconds")
    print(f"Each embedding shape: {seq_embeddings[0].shape}")
    
    # Test batch prediction
    print("\n--- Batch Processing of Multiple Images ---")
    batch_start_time = time.time()
    batch_future = blipv2_actor.predict_batch.remote(image_batch)
    batch_embeddings = ray.get(batch_future)
    batch_end_time = time.time()
    batch_time = batch_end_time - batch_start_time
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Embedding dimensionality: {batch_embeddings.shape[-1]}")
    print(f"Number of images in batch: {batch_embeddings.shape[0]}")
    print(f"Batch processing time for {len(image_batch)} images: {batch_time:.4f} seconds")
    print(f"Average time per image in batch: {batch_time/len(image_batch):.4f} seconds")
    
    # Calculate and display speedup
    speedup = seq_time / batch_time
    print(f"\n--- Performance Comparison ---")
    print(f"Speedup from batch processing: {speedup:.2f}x faster than sequential processing")
    
    # Plot
    try:
        labels = ['Single Image', 'Sequential (per image)', 'Batch (per image)']
        times = [single_time, seq_time/len(image_batch), batch_time/len(image_batch)]
        
        plt.figure(figsize=(10, 5))
        plt.bar(labels, times)
        plt.ylabel('Time (seconds)')
        plt.title('BLIP-2 Processing Time Comparison')
        plt.show()
        print(f"Performance comparison plot displayed")
    except Exception as e:
        print(f"Could not create performance plot: {e}")
    
    # Shutdown Ray
    ray.shutdown() 