import typer
import uvicorn
import glob 
import re
import asyncio
import time

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
import os
from rich.console import Console
from PIL import Image

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

# Update CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

def init_ray_with_env_vars(ray_tmp_dir: str = None):
    """Initialize Ray with proper environment variables"""
    if ray.is_initialized():
        ray.shutdown()
    
    if not ray.is_initialized():
        try:
        # Set environment variables for Ray
            os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable memory monitor to avoid fork issues
        
            # Initialize with version override
            ray.init(
                _temp_dir=ray_tmp_dir,
                ignore_reinit_error=False,
                # runtime_env={"pip": ["ray==2.40.0"]},  # Force specific Ray version
                # _system_config={
                #     "ignore_version_mismatch": True  # Ignore Python version mismatch
                # },
                # include_dashboard=False,  # Disable dashboard to reduce warnings
                # log_to_driver=False,  # Reduce logging
            )
            console.print("[green]Ray initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing Ray: {str(e)}[/red]")
            raise

class MolmoSAM2Service:
    def __init__(self,
                use_gpu: bool = True,
                verbose: bool = True,
                save_visualizations: bool = False,
                visualization_dir: str = os.path.join("src", "models", "segmentation", "save_tmp"),
                ray_tmp_dir: str = None,
                batch_size: int = 12
                ):
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.ray_tmp_dir = ray_tmp_dir
        self.verbose = verbose
        self.batch_size = batch_size

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This service requires GPU.")
            
        if self.verbose:
            console.print("\n[cyan]Initializing Molmo Pipeline...[/cyan]")
            
        vision_config = VisionConfig(max_batch_size=512)
        generation_config = {
            'do_sample': True,
            'top_p': 0.9,
            'temperature': 0.8,
            'max_new_tokens': 64
        }
        
        if self.verbose:
            console.print("[yellow]Loading Molmo model...[/yellow]")
            
        self.pipe = pipeline('/scratch/xubowen/model-test/models/models--allenai--Molmo-7B-D-0924/snapshots/1721478b71306fb7dc671176d5c204dc7a4d27d7', 
                           vision_config=vision_config,
                           generation_config=generation_config,
                           backend_config=TurbomindEngineConfig(
                           tp=tp,
                           cache_max_entry_count=0.15)
                           )
                           
        if self.verbose:
            console.print("[green]Successfully loaded Molmo model[/green]")

        # Initialize Ray if needed
        init_ray_with_env_vars(self.ray_tmp_dir)

        if self.verbose:
            console.print("\n[cyan]Initializing SAM Actor...[/cyan]")
            
        use_gpu = torch.cuda.is_available()
        sam_actor_options = {"num_gpus": 0.3} if use_gpu else {}
        self.sam_actor = SAM2Actor.options(**sam_actor_options).remote(use_gpu=use_gpu)
        
        if self.verbose:
            console.print("[green]SAM Actor initialized successfully[/green]")

    
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

    def predict_molmo(self, image_data_list, prompt_list, timings):
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
            console.print(f"Number of prompts (N): {len(prompt_list)}")
            console.print(f"Total combinations (M×N): {len(image_data_list) * len(prompt_list)}")
            console.print("\nPrompts:")
            for i, p in enumerate(prompt_list):
                console.print(f"  {i+1}. {p}")
        
        for img_idx, img_data in enumerate(image_data_list):
            img = decode_base64_image(img_data)
            for prompt_idx, p in enumerate(prompt_list):
                if self.verbose:
                    console.print(f"Pairing Image {img_idx+1} with Prompt {prompt_idx+1}: '{p}'")
                imgs_prompts.append((p, img))
                images.append(img)
        
        if self.verbose:
            console.print("\n[cyan]Processing Details:[/cyan]")
            console.print(f"Total image-prompt pairs: {len(imgs_prompts)}")
            console.print(f"Batch size: {self.batch_size}")
        
        with Timer(enable_print=True) as t:
            response = []
            # Process batches
            for i in range(0, len(imgs_prompts), self.batch_size):
                batch = imgs_prompts[i:i + self.batch_size]
                if self.verbose:
                    console.print(f"\nProcessing batch {i//self.batch_size + 1}/{(len(imgs_prompts)-1)//self.batch_size + 1}")
                    console.print(f"Batch size: {len(batch)}")
                response_batch = self.pipe(batch)
                response.extend(response_batch)
        timings[f"molmo_inference"] = t.get_elapsed_time()

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
            if self.verbose and i < 3:  # Show first 3 responses as examples
                console.print(f"\n[cyan]Response {i+1}:[/cyan]")
                console.print(f"Text: {resp.text}")
                console.print(f"Points detected: {len(result['points'])}")
            processed_response.append(result)
        
        if self.verbose:
            console.print(f"\n[green]Processing Summary:[/green]")
            console.print(f"Total processed responses: {len(processed_response)}")
            
        return processed_response

    async def _process_image(self, image_data: str) -> Image.Image:
        """Process base64 image data into PIL Image"""
        try:
            with Timer(enable_print=False) as t:
                image = decode_base64_image(image_data)
            if self.verbose:
                console.print(
                    f"Base64 decoding time: {t.get_elapsed_time():.3f}s", style="cyan"
                )
            return image
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    async def predict(
        self, image_data_list: str, prompts: list[str], render: bool = True
    ):
        """Main prediction method for multiple images and prompts
        Args:
            image_data_list: List of base64 encoded images
            prompts: List of prompts to apply to each image
            render: Whether to render and save visualization
        Returns:
            List of results, where each result contains predictions for one image-prompt pair
        """
        timings = {}
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_dir = os.path.join(self.visualization_dir, timestamp)
            os.makedirs(request_dir, exist_ok=True)
            
            if self.verbose:
                console.print("\n[cyan]Request Information:[/cyan]")
                console.print(f"Number of images: {len(image_data_list)}")
                console.print(f"Number of prompts: {len(prompts)}")
                console.print(f"Total combinations (M×N): {len(image_data_list) * len(prompts)}")
                console.print("\nPrompts:")
                for i, p in enumerate(prompts):
                    console.print(f"  {i+1}. {p}")
            
            # Get Molmo predictions
            with Timer(enable_print=True) as t:
                molmo_result = self.predict_molmo(image_data_list, prompts, timings)
            timings[f"molmo_total"] = t.get_elapsed_time()
            
            results = []
            with Timer(enable_print=False) as total_sam_vs_t:
                for img_idx, image_data in enumerate(image_data_list):
                    # Process image once
                    image = await self._process_image(image_data)
                    
                    # Save original image
                    if self.save_visualizations:
                        original_image_path = os.path.join(request_dir, f"original_image_{img_idx}.png")
                        image.save(original_image_path)
                        if self.verbose:
                            console.print(f"[green]Saved original image to: {original_image_path}[/green]")
                    
                    for prompt_idx, prompt in enumerate(prompts):
                        # Calculate the correct index in molmo_result
                        result_idx = img_idx * len(prompts) + prompt_idx
                        
                        if self.verbose:
                            console.print(f"\n[cyan]Processing Image {img_idx+1}, Prompt {prompt_idx+1}[/cyan]")
                            console.print(f"Prompt: '{prompt}'")
                        
                        with Timer(enable_print=False) as t:
                            # Get points from Molmo result
                            points = [[float(x), float(y)] for x, y in molmo_result[result_idx]["points"]]
                            
                            # Format base result
                            result = {
                                "image_index": img_idx,
                                "prompt_index": prompt_idx,
                                "prompt": prompts[prompt_idx],
                                "points": points,
                                "has_detection": len(points) > 0
                            }

                            if len(points) > 0:  # Only process with SAM if points were detected
                                if self.verbose:
                                    console.print(f"Found {len(points)} points, processing with SAM")
                                    
                                points_array = np.array(points).reshape(-1, 2)
                                
                                # Process with SAM
                                sam_result = ray.get(
                                    self.sam_actor.predict.remote(
                                        image,
                                        input=points_array.tolist(),
                                        type="point",
                                    )
                                )
                                timings[f"sam_inference_{result_idx}"] = t.get_elapsed_time()
                                
                                # Add SAM results
                                result["boxes"] = [
                                    [float(x) for x in box]
                                    for box in sam_result.boxes.data.tolist()
                                ] if hasattr(sam_result, "boxes") else None

                                if hasattr(sam_result, "masks"):
                                    masks = sam_result.masks.data
                                    result["masks"], size_info = process_masks(masks, self.verbose)

                                # Handle visualization only if points were detected
                                if render or self.save_visualizations:
                                    if self.verbose:
                                        console.print("\nGenerating visualizations:", style="cyan")
                                        console.print(f"Number of points: {len(points)}", style="cyan")
                                        if hasattr(sam_result, "masks"):
                                            console.print(f"Number of masks: {len(sam_result.masks)}", style="cyan")

                                    with Timer(enable_print=False) as img_visualization_t:
                                        points_path, boxes_path, masks_path = save_visualization(
                                            image=image,
                                            points=points,
                                            sam_result=sam_result,
                                            img_idx=img_idx,
                                            prompt_idx=prompt_idx,
                                            save_dir=request_dir,
                                        )
                                    timings[f"img_visualization_{result_idx}"] = img_visualization_t.get_elapsed_time()

                                    if render:
                                        result["points_image"] = encode_image(points_path)
                                        result["boxes_image"] = encode_image(boxes_path)
                                        result["masks_image"] = encode_image(masks_path)

                                    if self.verbose and self.save_visualizations:
                                        console.print("[green]Saved visualizations to:[/green]")
                                        console.print(f"Points: {points_path}")
                                        console.print(f"Boxes: {boxes_path}")
                                        console.print(f"Masks: {masks_path}")
                            else:
                                if self.verbose:
                                    console.print("[yellow]No points detected, skipping SAM processing[/yellow]")

                        results.append(result)

            # Print timing summary
            timings["total_sam_vs_time"] = total_sam_vs_t.get_elapsed_time()
            sam_total = sum(value for key, value in timings.items() if 'sam_inference' in key)
            vis_total = sum(value for key, value in timings.items() if 'img_visualization' in key)
            
            if self.verbose:
                console.print("\n[green]Performance Summary:[/green]")
                console.print(f"Molmo inference: {timings['molmo_total']:.3f}s", style="cyan")
                console.print(f"SAM total: {sam_total:.3f}s", style="cyan")
                console.print(f"Visualization total: {vis_total:.3f}s", style="cyan")
                console.print(f"Total results: {len(results)}", style="green")
            
            return results

        except Exception as e:
            console.print(f"[red]Error in predict: {str(e)}[/red]")
            import traceback
            console.print(traceback.format_exc(), style="red")
            raise


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
    # request_start_time = request.headers.get("X-Request-Start-Time")
    
    console.rule("[bold blue]New Request")
    console.print(f"[{Timer.get_current_time()}] Received new request", style="cyan")

    with Timer(enable_print=True, name="FastAPI total handler") as t_total:
        # Time JSON parsing
        with Timer(enable_print=True, name="Request JSON parsing") as t:
            data = await request.json()
            images = data.get("images", [])
            raw_prompts = data.get("prompts", [])
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

        console.print(f"json parsing time: {t.get_elapsed_time():.3f}s", style="cyan")
        # Time actual prediction
        with Timer(enable_print=True, name="Service prediction") as t:
            service.verbose = verbose
            task = asyncio.create_task(service.predict(images, prompts, data.get("render", True)))
        console.print(f"service prediction time: {t.get_elapsed_time():.3f}s", style="cyan")
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
        console.print(f"response serialization time: {t.get_elapsed_time():.3f}s", style="cyan")
    
    console.print(f"[{Timer.get_current_time()}] Request completed", style="cyan")
    console.print("=" * 50 + "\n", style="cyan")


def start_server(
    port: int = 7100,
    host: str = "0.0.0.0",  # Changed from implicit to explicit host
    use_gpu: bool = True,
    verbose: bool = True,
    save_visualizations: bool = True,
    visualization_dir: str = os.path.join("src", "models", "segmentation", "save_tmp"),
    ray_tmp_dir: str = "/scratch/wangyin/tmp/ray",
    batch_size: int = 12
):
    """Start the FastAPI server"""
    if save_visualizations:
        os.makedirs(visualization_dir, exist_ok=True)
        console.print(f"[yellow]Saving visualizations to: {visualization_dir}[/yellow]")

    global service
    service = MolmoSAM2Service(
        use_gpu=use_gpu,
        verbose=verbose,
        save_visualizations=save_visualizations,
        visualization_dir=visualization_dir,
        ray_tmp_dir=ray_tmp_dir,
        batch_size=batch_size
    )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Updated uvicorn configuration
    uvicorn.run(
        app,
        host=host,
        port=port,
        proxy_headers=True,  # Enable proxy headers
        forwarded_allow_ips="*",  # Trust forwarded IP headers
        access_log=True,  # Enable access logging
    )


if __name__ == "__main__":
    def cli(
        port: int = typer.Option(7100, help="Port number for the server"),
        host: str = typer.Option("0.0.0.0", help="Host IP to bind to"),  # Added host parameter
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
            None,
            help="Directory to save ray tmp",
        ),
        batch_size: int = typer.Option(
            12,
            help="Batch size for processing image-prompt pairs",
        ),
    ):
        """Start the MolmoSAM2 server with visualization options"""
        start_server(
            port=port,
            host=host,  # Added host parameter
            use_gpu=use_gpu,
            verbose=verbose,
            save_visualizations=save_visualizations,
            visualization_dir=visualization_dir,
            ray_tmp_dir=ray_tmp_dir,
            batch_size=batch_size
        )

    typer.run(cli)