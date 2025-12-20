import logging
import os
import re
from typing import List, Dict, Union

import ray
import torch
import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams

from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.vlms.molmo_quantized_actor import draw_points

@ray.remote
class BatchedMolmoActor:
    def __init__(self, use_gpu: bool = True, batch_size: int = 4):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        if not torch.cuda.is_available() or not use_gpu:
            raise RuntimeError("GPU is required for vLLM")
            
        # Get GPU device ID
        gpu_id = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        
        model_name = "allenai/Molmo-7B-D-0924"
        
        # Set environment variable to allow longer sequences
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        
        # Fix CUDA library path
        cuda_lib_path = "/usr/local/cuda/lib64"
        if os.path.exists(cuda_lib_path):
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # Initialize vLLM engine with corrected parameters
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.75,  # Lower utilization to avoid OOM
            max_num_batched_tokens=4096,
            max_num_seqs=batch_size,
            enforce_eager=True,
            tensor_parallel_size=1,  # Use single GPU
            quantization="awq",  # Use quantization to reduce memory usage
            max_model_len=4096,  # Set to model's max position embeddings
            device="cuda"  # Use CUDA device
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            stop=["<|endoftext|>"]
        )
        
        logging.info(f"vLLM engine initialized on CUDA device {gpu_id}")

    def _extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                continue
                
            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
        return all_points

    def batch_predict(
        self, 
        images: List[Image.Image], 
        prompts: List[str], 
        points: bool = True,
        use_scaled_input: bool = True
    ) -> List[Dict]:
        target_size = (576, 576)
        processed_images = []
        original_sizes = []
        
        # Preprocess images
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            original_sizes.append(img.size)
            if use_scaled_input:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            processed_images.append(img)

        results = []
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = processed_images[i:i + self.batch_size]
            batch_prompts = prompts[i:i + self.batch_size]
            batch_orig_sizes = original_sizes[i:i + self.batch_size]
            
            # Format prompts for vLLM
            formatted_prompts = []
            for img_idx, prompt in enumerate(batch_prompts):
                # Format prompt according to Molmo's requirements
                formatted_prompt = f"<image>Image {img_idx}</image> {prompt}"
                formatted_prompts.append(formatted_prompt)
            
            # Get batch predictions using vLLM
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            
            # Process each output
            for j, (output, orig_size) in enumerate(zip(outputs, batch_orig_sizes)):
                generated_text = output.outputs[0].text
                
                if points:
                    if use_scaled_input:
                        point_output = self._extract_points(
                            generated_text,
                            target_size[0],
                            target_size[1]
                        )
                        
                        # Scale points back
                        orig_w, orig_h = orig_size
                        scale_w = orig_w / target_size[0]
                        scale_h = orig_h / target_size[1]
                        scaled_points = []
                        for point in point_output:
                            scaled_point = np.array([
                                point[0] * scale_w,
                                point[1] * scale_h
                            ])
                            scaled_points.append(scaled_point)
                        final_points = scaled_points
                    else:
                        final_points = self._extract_points(
                            generated_text,
                            orig_size[0],
                            orig_size[1]
                        )
                    
                    results.append({
                        "text": generated_text,
                        "points": final_points
                    })
                else:
                    results.append({"text": generated_text})
                
        return results

@ray.remote(num_gpus=0.01)  # Minimal GPU allocation to enable CUDA
class BatchedSegmentationPipeline:
    def __init__(self, use_gpu: bool = True, batch_size: int = 4):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        # Fix: Use .remote() for actor instantiation with proper GPU allocation
        sam_actor_options = {"num_gpus": 0.48} if use_gpu else {}  # 48% GPU for SAM
        molmo_actor_options = {"num_gpus": 0.48} if use_gpu else {}  # 48% GPU for Molmo
        
        self.molmo_actor = BatchedMolmoActor.options(**molmo_actor_options).remote(
            use_gpu=use_gpu, 
            batch_size=batch_size
        )
        self.sam_actor = SAM2Actor.options(**sam_actor_options).remote(use_gpu=use_gpu)
        
    def process_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        output_dir: str
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        
        # Fix: Use ray.get() for actor method calls
        point_results = ray.get(self.molmo_actor.batch_predict.remote(images, prompts))
        
        output_paths = []
        for idx, (image, point_result) in enumerate(zip(images, point_results)):
            drawn_image = draw_points(image.copy(), point_result["points"])
            points_path = os.path.join(output_dir, f"points_{idx}.png")
            drawn_image.save(points_path)
            
            result = ray.get(self.sam_actor.predict.remote(
                drawn_image, 
                input=point_result["points"], 
                type="point"
            ))
            
            result_path = os.path.join(output_dir, f"segmentation_{idx}.png")
            result.save(result_path)
            output_paths.append(result_path)
            
        return output_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batched Molmo-SAM2 inference with vLLM')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--images', nargs='+', help='Paths to input images')
    parser.add_argument('--prompts', nargs='+', help='Prompts for each image')
    
    args = parser.parse_args()
    
    ray.init()
    
    # Use provided batch size or default
    batch_size = args.batch_size
    pipeline = BatchedSegmentationPipeline.remote(
        use_gpu=torch.cuda.is_available(), 
        batch_size=batch_size
    )
    
    # Use provided images or default test images
    if args.images:
        images = [Image.open(path) for path in args.images]
        prompts = args.prompts if args.prompts else ["Point at interesting objects."] * len(images)
    else:
        # Default test case
        images = [
            Image.open("tests/images/test_spot_table_cable_1.jpg"),
            Image.open("tests/images/test_img_cable_with_knots.jpg"),
        ]
        prompts = [
            "Point at the cups or table. Separate out different objects.",
            "Point at any knots in the cable."
        ]
    
    output_dir = os.path.join("src", "models", "tmp", "batch_results")
    results = ray.get(pipeline.process_batch.remote(images, prompts, output_dir))
    
    logging.info(f"Batch processing complete. Results saved in: {output_dir}")
    for path in results:
        print(f"Generated: {path}")
    
    ray.shutdown()
