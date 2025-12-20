"""
Usage:
1. Start the server:
   python -m src.models.segmentation.molmo_sam2_http_server

2. In another terminal, run the client:
   python -m src.models.segmentation.molmo_sam2_http_client "./tests/images/test_img_cable_with_knots.jpg" "Point at the cable knot"

Output files will be saved in ./src/models/tmp/:
- molmo_sam2_serve_points.png: Visualization of the detected points
- molmo_sam2_serve_segmentation.png: Final segmentation result
"""


import os
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import ray
import torch
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rich.console import Console
from fastapi.responses import JSONResponse

from retriever.models.segmentation.molmo_sam2_utils import (
    decode_base64_image,
    encode_image,
    format_timing_summary,
    process_masks,
    save_visualization,
)
from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.common_utils import Timer
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor

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

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MolmoSAM2Service:
    def __init__(
        self,
        use_gpu: bool = False,
        verbose: bool = True,
        save_visualizations: bool = False,
        visualization_dir: str = os.path.join("src", "models", "tmp"),
    ):
        self.verbose = verbose
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir

        if self.verbose:
            console.rule("[bold blue]Device Configuration")
            console.print(f"Using GPU: {use_gpu}", style="cyan")
            console.print(f"CUDA available: {torch.cuda.is_available()}", style="cyan")
            if torch.cuda.is_available():
                console.print(
                    f"CUDA device: {torch.cuda.get_device_name(0)}", style="green"
                )
                console.print("\n[yellow]Ray GPU Resources:[/yellow]")
                gpu_ids = ray.get_gpu_ids()
                for gpu_id in gpu_ids:
                    try:
                        # Convert string GPU ID to integer
                        gpu_idx = int(gpu_id)
                        gpu_name = torch.cuda.get_device_name(gpu_idx)
                        gpu_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (
                            1024**3
                        )  # Convert to GB
                        console.print(
                            f"GPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)", style="green"
                        )
                    except (ValueError, RuntimeError) as e:
                        console.print(f"Warning: Could not get info for GPU {gpu_id}: {e}", style="yellow")

        # Initialize the actors with GPU options if available
        molmo_actor_options = {"num_gpus": 0.48} if use_gpu else {}
        sam_actor_options = {"num_gpus": 0.48} if use_gpu else {}

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.molmo_actor = MolmoQuantizedActor.options(**molmo_actor_options).remote(
            use_gpu=use_gpu
        )
        self.sam_actor = SAM2Actor.options(**sam_actor_options).remote(use_gpu=use_gpu)

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
        self, image_data_list: list[str], prompts: list[str], render: bool = True
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
            # Create timestamped directory for this request
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_dir = os.path.join(self.visualization_dir, timestamp)
            os.makedirs(request_dir, exist_ok=True)
            if self.verbose:
                console.print(f"[yellow]Saving visualizations to: {request_dir}[/yellow]")

            if self.verbose:
                console.rule("[bold blue]Starting Batch Prediction")
                console.print(
                    f"Processing {len(image_data_list)} images with {len(prompts)} prompts each",
                    style="cyan",
                )
                console.print(
                    f"Total combinations: {len(image_data_list) * len(prompts)}",
                    style="cyan",
                )

            results = []
            # Process each image
            for img_idx, image_data in enumerate(image_data_list):
                if self.verbose:
                    console.print(
                        f"\nProcessing image {img_idx + 1}/{len(image_data_list)}",
                        style="cyan",
                    )

                # Decode image once per image
                with Timer(enable_print=False) as t:
                    image = await self._process_image(image_data)
                timings[f"base64_decode_{img_idx}"] = t.elapsed_time
                if self.verbose:
                    console.print(
                        f"Base64 decode time: {timings[f'base64_decode_{img_idx}']:.3f}s",
                        style="cyan",
                    )

                # Save original image
                if self.save_visualizations:
                    orig_filename = f"img{img_idx}_original_{timestamp}.png"
                    orig_path = os.path.join(request_dir, orig_filename)
                    image.save(orig_path)
                    if self.verbose:
                        console.print(f"[green]Saved original image to: {orig_path}[/green]")

                image_results = []
                # Apply each prompt to the current image
                for prompt_idx, prompt in enumerate(prompts):
                    if self.verbose:
                        console.print(
                            f"\nApplying prompt {prompt_idx + 1}/{len(prompts)}: {prompt}",
                            style="cyan",
                        )

                    # Get points from Molmo
                    if self.verbose:
                        console.print("Getting points from Molmo...", style="cyan")
                    with Timer(enable_print=False) as t:
                        molmo_result = ray.get(
                            self.molmo_actor.predict.remote(image, prompt, points=True)
                        )
                        if self.verbose:
                            device_info = ray.get(
                                self.molmo_actor.get_device_info.remote()
                            )
                            console.print(
                                f"Molmo model device: {device_info}", style="green"
                            )
                    timings[f"molmo_inference_{img_idx}_{prompt_idx}"] = t.elapsed_time
                    if self.verbose:
                        console.print(
                            f"Molmo inference: {timings[f'molmo_inference_{img_idx}_{prompt_idx}']:.3f}s",
                            style="cyan",
                        )

                    # Convert points
                    points = [[float(x), float(y)] for x, y in molmo_result["points"]]

                    # Skip further processing if no points detected
                    if not points:
                        if self.verbose:
                            console.print("[yellow]No points detected, skipping segmentation[/yellow]")
                        result = {
                            "image_index": img_idx,
                            "prompt_index": prompt_idx,
                            "prompt": prompt,
                            "points": [],
                            "boxes": None,
                            "masks": None
                        }
                        image_results.append(result)
                        continue

                    # Get segmentation from SAM
                    if self.verbose:
                        console.print("Getting segmentation from SAM...", style="cyan")
                    with Timer(enable_print=False) as t:
                        # Convert points to numpy array and ensure correct shape
                        points_array = np.array(points)
                        if len(points_array) == 0:
                            # Handle case where no points were detected
                            points_array = np.zeros(
                                (0, 2)
                            )  # Empty array with correct shape
                        else:
                            # Ensure points are in correct shape (N, 2)
                            points_array = points_array.reshape(-1, 2)

                        sam_result = ray.get(
                            self.sam_actor.predict.remote(
                                image,
                                input=points_array.tolist(),  # Convert back to list for serialization
                                type="point",
                            )
                        )
                        if self.verbose:
                            device_info = ray.get(
                                self.sam_actor.get_device_info.remote()
                            )
                            console.print(
                                f"SAM model device: {device_info}", style="green"
                            )

                            # Add simple object count display here
                            num_masks = (
                                len(sam_result.masks.data)
                                if hasattr(sam_result, "masks")
                                else 0
                            )
                            console.print(
                                f"[yellow]Objects detected for image {img_idx + 1}, prompt '{prompt}': {num_masks}[/yellow]"
                            )

                    timings[f"sam_inference_{img_idx}_{prompt_idx}"] = t.elapsed_time
                    if self.verbose:
                        console.print(
                            f"SAM inference: {timings[f'sam_inference_{img_idx}_{prompt_idx}']:.3f}s",
                            style="cyan",
                        )

                    # Prepare result for this image-prompt pair
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

                    # Handle masks with RLE compression
                    if hasattr(sam_result, "masks"):
                        masks = sam_result.masks.data
                        result["masks"], size_info = process_masks(masks, self.verbose)

                        if self.verbose:
                            console.print("\nMask debug info:", style="cyan")
                            console.print(
                                f"Mask shape: {size_info['shape']}", style="cyan"
                            )
                            console.print(
                                f"Mask dtype: {size_info['dtype']}", style="cyan"
                            )
                            console.print(
                                f"Number of masks: {size_info['num_masks']}",
                                style="cyan",
                            )
                            console.print(
                                f"Non-zero elements in first mask: {size_info['nonzero_first']}",
                                style="cyan",
                            )

                            console.print("\nSize analysis:", style="cyan")
                            console.print(
                                f"Original tensor size: {size_info['original_mb']:.2f}MB",
                                style="cyan",
                            )
                            console.print(
                                f"RLE counts size: {size_info['rle_counts_mb']:.2f}MB",
                                style="cyan",
                            )
                            console.print(
                                f"Final JSON size: {size_info['compressed_mb']:.2f}MB",
                                style="cyan",
                            )
                            console.print(
                                f"Compression ratio: {size_info['compression_ratio']:.1f}x",
                                style="cyan",
                            )

                    if self.verbose:
                        console.print("\nVisualization settings:", style="cyan")
                        console.print(f"Save visualizations: {self.save_visualizations}", style="cyan") 
                        console.print(f"Render: {render}", style="cyan")
                    if render or self.save_visualizations:
                        # Only save visualizations if points were detected
                        if points:
                            if self.verbose:
                                console.print("\nGenerating visualizations:", style="cyan")
                                console.print(f"Image index: {img_idx}", style="cyan")
                                console.print(f"Prompt index: {prompt_idx}", style="cyan")
                                console.print(f"Number of points: {len(points)}", style="cyan")
                                if hasattr(sam_result, "masks"):
                                    console.print(
                                        f"Number of masks: {len(sam_result.masks)}",
                                        style="cyan",
                                    )

                            points_path, boxes_path, masks_path = save_visualization(
                                image,
                                points,
                                sam_result,
                                img_idx,
                                prompt_idx,
                                save_dir=request_dir,
                            )

                            if render:
                                result["points_image"] = encode_image(points_path)
                                result["boxes_image"] = encode_image(boxes_path)
                                result["masks_image"] = encode_image(masks_path)

                            if self.verbose and self.save_visualizations:
                                console.print("[green]Saved visualizations to:[/green]")
                                console.print(f"Points: {points_path}")
                                console.print(f"Boxes: {boxes_path}")
                                console.print(f"Masks: {masks_path}")

                    image_results.append(result)

                results.extend(image_results)

            if self.verbose:
                console.print(
                    format_timing_summary(timings, len(image_data_list), prompts),
                    style="cyan",
                )

            return results

        except Exception as e:
            console.print(f"[red]Error in predict: {str(e)}[/red]", style="red")
            import traceback

            console.print(traceback.format_exc(), style="red")
            raise


def preprocess_prompt(prompt: str) -> str:
    """Ensure prompt starts with 'Point at the'"""
    prompt = prompt.strip()
    if "point at" not in prompt.lower():
        console.print(
            f"[yellow]Info: Prompt '{prompt}' doesn't start with 'Point at'. "
            "Adding prefix...[/yellow]"
        )
        # If it doesn't start with "Point at the", add it
        if prompt.lower().startswith("the "):
            prompt = f"Point at {prompt}"
        else:
            prompt = f"Point at the {prompt}"
            
    # NOTE: add additional prompt to the end of the prompt
    prompt = f"{prompt}. (Only provide points for reasonable movable objects.)"
    return prompt


@app.post("/molmo_sam2_service")
async def predict(
    request: Request,
    verbose: bool = Query(True, description="Enable verbose timing and device information"),
):
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

        # Time actual prediction
        with Timer(enable_print=True, name="Service prediction") as t:
            service.verbose = verbose
            response = await service.predict(images, prompts, data.get("render", True))

        # Time response serialization
        with Timer(enable_print=True, name="Response serialization") as t:
            json_response = JSONResponse(content=response)
            console.print(f"Response size: {len(str(response)) / 1024 / 1024:.2f} MB", style="cyan")

    console.print(f"[{Timer.get_current_time()}] Request completed", style="cyan")
    console.print("=" * 50 + "\n", style="cyan")
    return json_response


def start_server(
    port: int = 8100,
    use_gpu: bool = True,
    verbose: bool = True,
    save_visualizations: bool = True,
    visualization_dir: str = os.path.join("src", "models", "tmp"),
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
    )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
        
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import typer

    def cli(
        port: int = typer.Option(8100, help="Port number for the server"),
        use_gpu: bool = typer.Option(True, help="Whether to use GPU"),
        verbose: bool = typer.Option(True, help="Show verbose output"),
        save_visualizations: bool = typer.Option(
            True, help="Save visualization images"
        ),
        visualization_dir: str = typer.Option(
            os.path.join("src", "models", "tmp"),
            help="Directory to save visualizations",
        ),
    ):
        """Start the MolmoSAM2 server with visualization options"""
        start_server(
            port=port,
            use_gpu=use_gpu,
            verbose=verbose,
            save_visualizations=save_visualizations,
            visualization_dir=visualization_dir,
        )

    typer.run(cli)
