import typer
import uvicorn
import glob 
import re
import asyncio
import time
import os
from typing import List, Tuple

# Set environment variable to handle tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lmdeploy import pipeline, VisionConfig, TurbomindEngineConfig 
from lmdeploy.vl import load_image

from contextlib import asynccontextmanager
from datetime import datetime
import nest_asyncio 

import ray
import numpy as np
import torch
from rich.console import Console
from PIL import Image
from rich.table import Table
from rich import box

from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.common_utils import Timer
from retriever.models.segmentation.molmo_sam2_utils import (
    decode_base64_image,
    encode_image,
    format_timing_summary,
    process_masks,
    save_visualization,
)


nest_asyncio.apply()

service = None
tp = 1 # only used for very large model. By default is 1.

# Initialize console globally
console = Console()

# Create a global service instance
service = None

def init_ray_with_env_vars(ray_tmp_dir: str):
    """Initialize Ray with proper environment variables"""
    if not ray.is_initialized():
        # Set environment variables for Ray
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable memory monitor to avoid fork issues
        
        console.print("[cyan]Initializing Ray...[/cyan]")
        ray.init(
            _temp_dir=ray_tmp_dir,
            ignore_reinit_error=True,
            include_dashboard=False,  # Disable dashboard to reduce warnings
            log_to_driver=False,  # Reduce logging
        )
        console.print("[green]Ray initialized successfully[/green]")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Don't initialize service here
    yield

    # Shutdown: Clean up resources
    console.print("[bold red]Shutting down MolmoSAM2 service...[/bold red]")
    if ray.is_initialized():
        ray.shutdown()


# Initialize FastAPI with lifespan handler
app = FastAPI(lifespan=lifespan)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@ray.remote
class MolmoActor:
    def __init__(self, gpu_id: int):
        """Initialize Molmo pipeline on specific GPU"""
        # Create local console instance
        self.console = Console()
        
        try:
            self.console.print(f"[yellow]Starting Molmo Actor initialization on GPU {gpu_id}...[/yellow]")
            
            # Set CUDA device
            torch.cuda.set_device(gpu_id)
            self.gpu_id = gpu_id
            
            # Print GPU info
            self.console.print(f"[cyan]Current CUDA device: {torch.cuda.current_device()}[/cyan]")
            self.console.print(f"[cyan]Device name: {torch.cuda.get_device_name(gpu_id)}[/cyan]")
            self.console.print(f"[cyan]Memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB[/cyan]")
            
            self.console.print("[yellow]Setting up Molmo pipeline configuration...[/yellow]")
            vision_config = VisionConfig(max_batch_size=512)
            generation_config = {
                'do_sample': True,
                'top_p': 0.9,
                'temperature': 0.8,
                'max_new_tokens': 64
            }
            
            self.console.print("[yellow]Loading Molmo model...[/yellow]")
            self.pipe = pipeline('/scratch/xubowen/model-test/models/models--allenai--Molmo-7B-D-0924/snapshots/1721478b71306fb7dc671176d5c204dc7a4d27d7', 
                               vision_config=vision_config,
                               generation_config=generation_config,
                               backend_config=TurbomindEngineConfig(
                               tp=tp,
                               cache_max_entry_count=0.15)
                               )
            self.console.print(f"[green]Successfully loaded Molmo model on GPU {gpu_id}[/green]")
            
            # Test the pipeline with a small dummy input
            self.console.print("[yellow]Running test inference...[/yellow]")
            dummy_image = Image.new('RGB', (224, 224), color='white')
            dummy_prompt = "Point at the object"
            self.pipe([(dummy_prompt, dummy_image)])
            self.console.print(f"[green]Successfully verified Molmo Actor on GPU {gpu_id} with test input[/green]")
            
            # Print final GPU state
            self.console.print(f"[cyan]Final memory allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB[/cyan]")
            
        except Exception as e:
            self.console.print(f"[red]Error initializing Molmo Actor on GPU {gpu_id}:[/red]")
            self.console.print(f"[red]{str(e)}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            raise

    def process_batch(self, batch):
        """Process a batch of image-prompt pairs"""
        try:
            # Verify we're on the correct GPU
            current_device = torch.cuda.current_device()
            if current_device != self.gpu_id:
                self.console.print(f"[yellow]Warning: Current device {current_device} != assigned GPU {self.gpu_id}, setting correct device[/yellow]")
                torch.cuda.set_device(self.gpu_id)
            return self.pipe(batch)
        except Exception as e:
            self.console.print(f"[red]Error in process_batch on GPU {self.gpu_id}: {str(e)}[/red]")
            raise
        
    def get_gpu_info(self):
        """Get GPU information for this actor"""
        try:
            return {
                "gpu_id": self.gpu_id,
                "device_name": torch.cuda.get_device_name(self.gpu_id),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated(self.gpu_id) / 1024**2
            }
        except Exception as e:
            self.console.print(f"[red]Error getting GPU info: {str(e)}[/red]")
            raise

class MolmoSAM2Service:
    def __init__(self,
                use_gpu: bool = True,
                verbose: bool = True,
                save_visualizations: bool = False,
                visualization_dir: str = os.path.join("src", "models", "segmentation", "save_tmp"),
                ray_tmp_dir: str = "/scratch/wangyin/tmp/ray",
                gpu_ids: List[int] = None,  # Specific GPU IDs to use
                num_gpus: int = None,  # Number of GPUs to use (auto-select least used)
                gpu_memory_fraction: float = 0.25  # Memory fraction per actor
                ):
        # Assert GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This service requires GPU.")
        
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.ray_tmp_dir = ray_tmp_dir
        self.verbose = verbose
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # GPU selection logic
        if gpu_ids is not None:
            # Use specified GPU IDs
            self.gpu_ids = [id for id in gpu_ids if id < torch.cuda.device_count()]
            if len(self.gpu_ids) != len(gpu_ids):
                console.print(f"[yellow]Warning: Some specified GPU IDs are invalid. Using {self.gpu_ids}[/yellow]")
        elif num_gpus is not None:
            # Auto-select least used GPUs
            self.gpu_ids = self._select_least_used_gpus(num_gpus)
        else:
            # Default to all available GPUs
            self.gpu_ids = list(range(torch.cuda.device_count()))
        
        # Assert we have at least one GPU
        if not self.gpu_ids:
            raise RuntimeError("No valid GPUs found. This service requires at least one GPU.")
        
        if self.verbose:
            console.print(f"[green]Using GPUs: {self.gpu_ids}[/green]")
            
        # Initialize Molmo actors for each GPU with increased timeout
        self.molmo_actors = []
        for gpu_id in self.gpu_ids:
            try:
                actor_options = {
                    "num_gpus": self.gpu_memory_fraction,
                    "num_cpus": 1,
                    "resources": {f"GPU_{gpu_id}": 1}  # Ensure actor uses specific GPU
                }
                if self.verbose:
                    console.print(f"\n[cyan]Creating Molmo Actor for GPU {gpu_id}[/cyan]")
                    console.print(f"Actor options: {actor_options}")
                
                actor = MolmoActor.options(**actor_options).remote(gpu_id)
                
                # Wait for actor to initialize with increased timeout
                try:
                    if self.verbose:
                        console.print(f"[yellow]Waiting for Molmo Actor on GPU {gpu_id} to initialize (timeout: 300s)...[/yellow]")
                    gpu_info = ray.get(actor.get_gpu_info.remote(), timeout=300)  # 5 minute timeout
                    self.molmo_actors.append(actor)
                    if self.verbose:
                        console.print(f"[green]Successfully created and verified Molmo Actor:[/green]")
                        console.print(f"[green]GPU ID: {gpu_info['gpu_id']}[/green]")
                        console.print(f"[green]Device: {gpu_info['device_name']}[/green]")
                        console.print(f"[green]Memory Used: {gpu_info['memory_allocated_mb']:.2f} MB[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to initialize Molmo Actor on GPU {gpu_id}: {str(e)}[/red]")
                    raise
                    
            except Exception as e:
                console.print(f"[red]Error creating Molmo Actor on GPU {gpu_id}: {str(e)}[/red]")
                raise

        # Initialize SAM actors for selected GPUs with timeout
        self.sam_actors = []
        for gpu_id in self.gpu_ids:
            try:
                sam_actor_options = {
                    "num_gpus": self.gpu_memory_fraction,
                    "num_cpus": 1,
                    "resources": {f"GPU_{gpu_id}": 1}  # Ensure actor uses specific GPU
                }
                actor = SAM2Actor.options(**sam_actor_options).remote(device=f"cuda:{gpu_id}")  # Specify GPU device
                
                # Wait for actor to initialize with timeout
                try:
                    ray.get(actor.get_device_info.remote(), timeout=60)  # 60 second timeout
                    self.sam_actors.append(actor)
                    if self.verbose:
                        console.print(f"[green]Successfully created and verified SAM Actor on GPU {gpu_id}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to initialize SAM Actor on GPU {gpu_id}: {str(e)}[/red]")
                    raise
                    
            except Exception as e:
                console.print(f"[red]Error creating SAM Actor on GPU {gpu_id}: {str(e)}[/red]")
                raise

    def _select_least_used_gpus(self, num_gpus: int) -> List[int]:
        """Select the least used GPUs based on current memory usage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Get memory usage for each GPU
            gpu_memory_usage = []
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = info.used / info.total
                gpu_memory_usage.append((i, memory_used))
            
            # Sort by memory usage and select the least used GPUs
            gpu_memory_usage.sort(key=lambda x: x[1])
            selected_gpus = [gpu[0] for gpu in gpu_memory_usage[:num_gpus]]
            
            if self.verbose:
                console.print("\n[cyan]GPU Memory Usage:[/cyan]")
                for gpu_id, usage in gpu_memory_usage:
                    status = "SELECTED" if gpu_id in selected_gpus else "NOT SELECTED"
                    console.print(f"GPU {gpu_id}: {usage*100:.1f}% used - {status}")
            
            return selected_gpus
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not query GPU memory usage: {e}[/yellow]")
            console.print("[yellow]Falling back to first N GPUs[/yellow]")
            return list(range(min(num_gpus, torch.cuda.device_count())))

    def _distribute_work(self, total_items: int) -> List[Tuple[int, int]]:
        """Distribute work across available GPUs/actors"""
        num_workers = len(self.sam_actors)
        base_items_per_worker = total_items // num_workers
        extra_items = total_items % num_workers
        
        allocations = []
        start_idx = 0
        for i in range(num_workers):
            current_allocation = base_items_per_worker + (1 if i < extra_items else 0)
            end_idx = start_idx + current_allocation
            if current_allocation > 0:
                allocations.append((start_idx, end_idx))
            start_idx = end_idx
        return allocations

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

    def predict_molmo(self, image_data_list, prompt, timings, use_cross_product=True):
        """Process images through Molmo pipeline using multiple actors
        Args:
            image_data_list: List of base64 encoded images
            prompt: List of prompts
            timings: Dict to store timing information
        Returns:
            List of processed responses
        """
        imgs_prompts = []
        images = []
        
        # Generate all M×N combinations
        if self.verbose:
            console.print("\n[cyan]Generating Image-Prompt Combinations:[/cyan]")
            console.print(f"Number of images (M): {len(image_data_list)}")
            console.print(f"Number of prompts (N): {len(prompt)}")
            console.print(f"Total combinations (M×N): {len(image_data_list) * len(prompt)}")
            console.print("\nPrompts:")
            for i, p in enumerate(prompt):
                console.print(f"  {i+1}. {p}")
        
        for img_idx, img_data in enumerate(image_data_list):
            img = decode_base64_image(img_data)
            for prompt_idx, p in enumerate(prompt):
                if self.verbose:
                    console.print(f"Pairing Image {img_idx+1} with Prompt {prompt_idx+1}: '{p}'")
                imgs_prompts.append((p, img))
                images.append(img)
        
        if self.verbose:
            console.print("\n[cyan]Processing Details:[/cyan]")
            console.print(f"Total image-prompt pairs: {len(imgs_prompts)}")
            console.print(f"Batch size: 8")
        
        with Timer(enable_print=True) as t:
            batch_size = 8
            response = []
            
            # Distribute batches across GPUs
            gpu_allocations = self._distribute_work(len(imgs_prompts))
            
            # Create tasks for each GPU's batch
            tasks = []
            for gpu_id, (start_idx, end_idx) in enumerate(gpu_allocations):
                # Get batches for this GPU
                gpu_pairs = imgs_prompts[start_idx:end_idx]
                
                # Process batches on this GPU's actor
                for i in range(0, len(gpu_pairs), batch_size):
                    batch = gpu_pairs[i:i + batch_size]
                    if self.verbose:
                        console.print(f"\n[cyan]Processing batch on GPU {self.gpu_ids[gpu_id]}: {i//batch_size + 1}/{(len(gpu_pairs)-1)//batch_size + 1}[/cyan]")
                        console.print(f"Batch size: {len(batch)}")
                    task = self.molmo_actors[gpu_id].process_batch.remote(batch)
                    tasks.append(task)
            
            # Wait for all tasks to complete
            responses = ray.get(tasks)
            for batch_response in responses:
                response.extend(batch_response)
                
        timings["molmo_inference"] = t.elapsed_time

        if self.verbose:
            console.print(f"\n[green]Molmo Results:[/green]")
            console.print(f"Total responses: {len(response)}")
        
        processed_response = []
        for i, resp in enumerate(response):
            image_w, image_h = images[i].size
            result = {
                "text": resp.text,
                "points": self._extract_points(resp.text, image_w, image_h)
            }
            processed_response.append(result)
            
            if self.verbose and i < 3:  # Show first 3 responses as examples
                console.print(f"\n[cyan]Response {i+1}:[/cyan]")
                console.print(f"Text: {resp.text}")
                console.print(f"Points detected: {len(result['points'])}")
        
        if self.verbose:
            console.print(f"\n[green]Processing Summary:[/green]")
            console.print(f"Total processed responses: {len(processed_response)}")
            console.print(f"Molmo inference time: {t.elapsed_time:.3f}s")
        return processed_response

    async def _process_image(self, image_data: str) -> Image.Image:
        """Process base64 image data into PIL Image"""
        try:
            with Timer(enable_print=False) as t:
                image = decode_base64_image(image_data)
            if self.verbose:
                console.print(
                    f"Base64 decoding time: {t.elapsed_time:.3f}s", style="cyan"
                )
            return image
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    async def predict(
        self, image_data_list: List[str], prompts: List[str], render: bool = True, use_cross_product: bool = True
    ):
        """Main prediction method for multiple images and prompts
        Args:
            image_data_list: List of base64 encoded images
            prompts: List of prompts
            render: Whether to render and save visualization
            use_cross_product: If True, generate all combinations of images and prompts (M×N)
        Returns:
            List of results, where each result contains predictions for one image-prompt pair
        """
        timings = {}
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_dir = os.path.join(self.visualization_dir, timestamp)
            os.makedirs(request_dir, exist_ok=True)
            
            if self.verbose:
                # Create request info table
                info_table = Table(title="Request Information", box=box.ROUNDED)
                info_table.add_column("Parameter", style="cyan")
                info_table.add_column("Value", style="green")
                
                info_table.add_row("Timestamp", timestamp)
                info_table.add_row("Images", str(len(image_data_list)))
                info_table.add_row("Prompts", str(len(prompts)))
                info_table.add_row("Mode", "Cross Product" if use_cross_product else "Matching Pairs")
                info_table.add_row("Expected combinations", 
                    str(len(image_data_list) * len(prompts) if use_cross_product else len(image_data_list)))
                
                console.print(info_table)
                
                # Create prompts table
                prompts_table = Table(title="Prompts", box=box.ROUNDED)
                prompts_table.add_column("#", style="cyan")
                prompts_table.add_column("Prompt", style="green")
                
                for i, p in enumerate(prompts, 1):
                    prompts_table.add_row(str(i), p)
                
                console.print(prompts_table)
            
            # Get Molmo predictions
            with Timer(enable_print=True) as t:
                molmo_result = self.predict_molmo(image_data_list, prompts, timings, use_cross_product)
            timings["molmo_total"] = t.elapsed_time
            
            # Process batches in parallel using Ray
            results = []
            with Timer(enable_print=True, name="SAM Processing") as total_sam_t:
                tasks = []
                gpu_allocations = self._distribute_work(len(image_data_list))
                
                if self.verbose:
                    # Create GPU allocation table
                    gpu_table = Table(title="GPU Allocations", box=box.ROUNDED)
                    gpu_table.add_column("GPU ID", style="cyan")
                    gpu_table.add_column("Images", style="green")
                    gpu_table.add_column("Indices", style="yellow")
                    
                    for gpu_id, (start, end) in enumerate(gpu_allocations):
                        gpu_table.add_row(
                            str(self.gpu_ids[gpu_id]),
                            str(end-start),
                            f"{start}-{end}"
                        )
                    
                    console.print(gpu_table)
                
                for gpu_id, (start_idx, end_idx) in enumerate(gpu_allocations):
                    batch_images = image_data_list[start_idx:end_idx]
                    batch_molmo_results = molmo_result[start_idx:end_idx]
                    batch_prompts = prompts[start_idx:end_idx]
                    
                    task = asyncio.create_task(
                        self._process_batch(
                            gpu_id, batch_images, batch_molmo_results,
                            batch_prompts, start_idx, render, request_dir, timings
                        )
                    )
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks)
                for batch_result in batch_results:
                    results.extend(batch_result)
            
            timings["sam_total"] = total_sam_t.elapsed_time
            
            if self.verbose:
                # Create timing table
                timing_table = Table(title="Performance Summary", box=box.DOUBLE)
                timing_table.add_column("Stage", style="cyan")
                timing_table.add_column("Time (s)", justify="right", style="green")
                
                timing_table.add_row("Molmo Inference", f"{timings['molmo_total']:.3f}")
                timing_table.add_row("SAM Processing", f"{timings['sam_total']:.3f}")
                if "vis_total" in timings:
                    timing_table.add_row("Visualization", f"{timings['vis_total']:.3f}")
                timing_table.add_row("Total Processing", 
                    f"{timings['molmo_total'] + timings['sam_total']:.3f}", 
                    style="bold green")
                
                console.print(timing_table)
                
                # Create results summary table
                summary_table = Table(title="Results Summary", box=box.ROUNDED)
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="green")
                
                summary_table.add_row("Total results generated", str(len(results)))
                summary_table.add_row("Average time per image", 
                    f"{(timings['molmo_total'] + timings['sam_total'])/len(image_data_list):.3f}s")
                
                console.print(summary_table)
            
            return results

        except Exception as e:
            console.print(f"[red]Error in predict: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc(), style="red")
            raise

    async def _process_batch(
        self,
        gpu_id: int,
        batch_images: List[str],
        batch_molmo_results: List[dict],
        batch_prompts: List[str],
        start_idx: int,
        render: bool,
        request_dir: str,
        timings: dict
    ) -> List[dict]:
        """Process a batch of images on a specific GPU"""
        batch_results = []
        
        for i, (image_data, molmo_res, prompt) in enumerate(
            zip(batch_images, batch_molmo_results, batch_prompts)
        ):
            img_idx = start_idx + i
            prompt_idx = img_idx  # For cross-product case, this might need adjustment
            
            # Process image
            image = await self._process_image(image_data)
            
            # Extract points and process with SAM
            with Timer(enable_print=False) as t:
                points = [[float(x), float(y)] for x, y in molmo_res["points"]]
                points_array = np.array(points)
                if len(points_array) == 0:
                    points_array = np.zeros((0, 2))
                else:
                    points_array = points_array.reshape(-1, 2)
                
                # Use the assigned GPU's SAM actor
                sam_result = ray.get(
                    self.sam_actors[gpu_id].predict.remote(
                        image,
                        input=points_array.tolist(),
                        type="point",
                    )
                )
            
            timings[f"sam_inference_{img_idx}"] = t.elapsed_time
            
            # Format result
            result = {
                "image_index": img_idx,
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "points": points,
                "boxes": [
                    [float(x) for x in box]
                    for box in sam_result.boxes.data.tolist()
                ]
                if hasattr(sam_result, "boxes")
                else None,
            }
            
            if hasattr(sam_result, "masks"):
                masks = sam_result.masks.data
                result["masks"], size_info = process_masks(masks, self.verbose)
            
            # Handle visualization if needed
            if render or self.save_visualizations:
                if self.verbose:
                    console.print(f"\nGenerating visualizations for batch {gpu_id}, image {i}", style="cyan")
                    if hasattr(sam_result, "masks"):
                        console.print(f"Number of masks: {len(sam_result.masks)}", style="cyan")
                
                with Timer(enable_print=False) as img_visualization_t:
                    points_path, boxes_path, masks_path = save_visualization(
                        image,
                        f"image_{img_idx}",  # Use index as filename since we don't have paths
                        points,
                        sam_result,
                        img_idx,
                        prompt_idx,
                        save_dir=request_dir
                    )
                timings[f"img_visualization_{img_idx}"] = img_visualization_t.elapsed_time
                
                if render:
                    result["points_image"] = encode_image(points_path)
                    result["boxes_image"] = encode_image(boxes_path)
                    result["masks_image"] = encode_image(masks_path)
                
                if self.verbose and self.save_visualizations:
                    console.print("[green]Saved visualizations to:[/green]")
                    console.print(f"Points: {points_path}")
                    console.print(f"Boxes: {boxes_path}")
                    console.print(f"Masks: {masks_path}")
            
            batch_results.append(result)
        
        return batch_results


def preprocess_prompt(prompt: str) -> str:
    """Ensure prompt starts with 'Point at the'"""
    prompt = prompt.strip()
    if "point at" not in prompt.lower():
        console.print(
            f"[yellow]Warning: Prompt '{prompt}' doesn't start with 'Point at'. "
            "Adding prefix...[/yellow]"
        )
        # If it doesn't start with "Point at the", add it
        if "the " in prompt.lower():
            prompt = f"Point at {prompt}"
        else:
            prompt = f"Point at the {prompt}"
            
    # NOTE: add additional prompt to the end of the prompt
    # prompt = f"{prompt}. (Only provide points for reasonable movable objects.)"
    prompt = f"{prompt}"
    return prompt


@app.post("/molmo_sam2_service")
async def predict(request: Request, verbose: bool = Query(True)):
    console.rule("[bold blue]New Request")
    console.print(f"[{Timer.get_current_time()}] Received new request", style="cyan")

    with Timer(enable_print=True, name="FastAPI total handler") as t_total:
        # Time JSON parsing
        with Timer(enable_print=True, name="Request JSON parsing") as t:
            data = await request.json()
            images = data.get("images", [])
            raw_prompts = data.get("prompts", [])
            image_paths = data.get("image_paths", [])
            prompts = [preprocess_prompt(p) for p in raw_prompts]

            if not images or not prompts:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing required fields: images and/or prompts"},
                )

            console.print(f"Number of images: {len(images)}", style="cyan")
            console.print(f"Number of prompts: {len(prompts)}", style="cyan")
            console.print(f"Total combinations: {len(images) * len(prompts)}", style="cyan")
            console.print(f"Total input size: {sum(len(img) for img in images) / 1024 / 1024:.2f} MB", style="cyan")

        console.print(f"json parsing time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time actual prediction
        with Timer(enable_print=True, name="Service prediction") as t:
            service.verbose = verbose
            task = asyncio.create_task(service.predict(images, prompts, data.get("render", True), data.get("use_cross_product", True)))
            
        console.print(f"service prediction time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time response serialization
        with Timer(enable_print=True, name="Response serialization") as t:
            try:
                response = await asyncio.wait_for(task, timeout=120)
                console.print(f"Response size: {len(str(response)) / 1024 / 1024:.2f} MB", style="cyan")
                return JSONResponse(content=response)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=408,
                    content={"error": "Request timeout"}
                )
        console.print(f"response serialization time: {t.elapsed_time:.3f}s", style="cyan")
    
    console.print(f"[{Timer.get_current_time()}] Request completed", style="cyan")
    console.print("=" * 50 + "\n", style="cyan")


def start_server(
    port: int = 7010,
    use_gpu: bool = True,
    verbose: bool = True,
    save_visualizations: bool = True,
    visualization_dir: str = os.path.join("src", "models", "segmentation", "save_tmp"),
    ray_tmp_dir: str = "/scratch/wangyin/tmp/ray",
    gpu_ids: List[int] = None,
    num_gpus: int = None,
    gpu_memory_fraction: float = 0.25
):
    """Start the FastAPI server"""
    if save_visualizations:
        os.makedirs(visualization_dir, exist_ok=True)
        console.print(f"[yellow]Saving visualizations to: {visualization_dir}[/yellow]")

    # Initialize Ray with our custom function
    init_ray_with_env_vars(ray_tmp_dir)

    global service
    service = MolmoSAM2Service(
        use_gpu=use_gpu,
        verbose=verbose,
        save_visualizations=save_visualizations,
        visualization_dir=visualization_dir,
        ray_tmp_dir=ray_tmp_dir,
        gpu_ids=gpu_ids,
        num_gpus=num_gpus,
        gpu_memory_fraction=gpu_memory_fraction
    )
        
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")  # Reduce uvicorn logging


if __name__ == "__main__":
    def cli(
        port: int = typer.Option(7010, help="Port number for the server"),
        use_gpu: bool = typer.Option(True, help="Whether to use GPU"),
        verbose: bool = typer.Option(True, help="Show verbose output"),
        save_visualizations: bool = typer.Option(
            True, help="Save visualization images"
        ),
        visualization_dir: str = typer.Option(
            os.path.join("src", "models", "segmentation", "visualizations_save_tmp"),
            help="Directory to save visualizations",
        ),
        ray_tmp_dir: str = typer.Option(
            "/scratch/wangyin/tmp/ray",
            help="Directory to save ray tmp",
        ),
        gpu_ids: str = typer.Option(
            None,
            help="Comma-separated list of specific GPU IDs to use (e.g. '0,1,2')",
        ),
        num_gpus: int = typer.Option(
            None,
            help="Number of GPUs to use (auto-selects least used GPUs)",
        ),
        gpu_memory_fraction: float = typer.Option(
            0.25,
            help="Fraction of GPU memory to allocate per SAM actor (0.0-1.0)",
        ),
    ):
        """Start the MolmoSAM2 server with visualization options"""
        # Parse GPU IDs if provided
        parsed_gpu_ids = None
        if gpu_ids is not None:
            try:
                parsed_gpu_ids = [int(id.strip()) for id in gpu_ids.split(",")]
            except ValueError:
                console.print("[red]Error: Invalid GPU IDs format. Use comma-separated integers (e.g. '0,1,2')[/red]")
                raise typer.Exit(1)
        
        start_server(
            port=port,
            use_gpu=use_gpu,
            verbose=verbose,
            save_visualizations=save_visualizations,
            visualization_dir=visualization_dir,
            ray_tmp_dir=ray_tmp_dir,
            gpu_ids=parsed_gpu_ids,
            num_gpus=num_gpus,
            gpu_memory_fraction=gpu_memory_fraction
        )

    typer.run(cli)