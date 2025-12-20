"""Pointing Gemini SAM2 service with HTTP server."""

import os
import base64
import io
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import time
import json
import logging

import ray
import torch
from PIL import Image
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.api_models.utils.gemini_parsing import (
    parse_gemini_point_response,
    parse_gemini_detection_response,
    PointData,
    BoxData,
    visualize_results,
)
from retriever.models.api_models.utils.google_utils import GeminiClient, GEMINI_MODEL_NAME
from retriever.models.common_utils import Timer

# Initialize console globally
console = Console()
install(show_locals=True)

def decode_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app."""
    global service
    # Initialize service on startup
    service = PointingGeminiSAM2Service()
    yield
    # Cleanup on shutdown
    if service is not None:
        service.cleanup()

app.router.lifespan_context = lifespan

class PointingGeminiSAM2Service:
    """Pointing Gemini SAM2 service."""

    def __init__(
        self,
        use_gpu: bool = True,
        verbose: bool = True,
        save_visualizations: bool = False,
        visualization_dir: str = os.path.join("src", "models", "tmp"),
        point_radius: int = 10,
        box_width: int = 2,
        mask_alpha: float = 0.5,
    ):
        """Initialize the service."""
        # TODO: Add Gemini segmentation support when available
        # TODO: Add support for Gemini's multimodal capabilities
        self.verbose = verbose
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.point_radius = point_radius
        self.box_width = box_width
        self.mask_alpha = mask_alpha

        if self.verbose:
            console.rule("[bold blue]Device Configuration")
            console.print(f"Using GPU: {use_gpu}", style="cyan")
            console.print(f"CUDA available: {torch.cuda.is_available()}", style="cyan")
            if torch.cuda.is_available():
                console.print(f"CUDA device: {torch.cuda.get_device_name(0)}", style="green")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available. This service requires a GPU.")
            ray.init(num_gpus=1)

        # Initialize SAM2 actor
        self.sam_actor = SAM2Actor.options(num_gpus=1).remote()  # type: ignore
        
        # Initialize Gemini client
        self.engine = GeminiClient()
        if self.engine is None:
            raise ValueError(f"Failed to initialize engine for model {GEMINI_MODEL_NAME}")
        
        self.model_name = GEMINI_MODEL_NAME

        # Create visualization directory if needed
        if self.save_visualizations:
            os.makedirs(self.visualization_dir, exist_ok=True)
            console.print(f"Saving visualizations to: {self.visualization_dir}", style="cyan")

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'sam_actor'):
            ray.kill(self.sam_actor)

    def _format_prompt(self, object_names: Union[str, List[str]], detection: bool = False) -> str:
        """Format the prompt for Gemini with the exact required format."""
        if isinstance(object_names, str):
            object_names = [object_names]
            
        objects_str = ", ".join(object_names)
        
        if detection:
            return (
                f"Detect {objects_str}, with no more than 20 items. "
                "Output a json list where each entry contains the 2D bounding box in \"box_2d\" and a text label in \"label\". "
                "The boxes are in [y_min, x_min, y_max, x_max] format normalized to 0-1000."
            )
        else:
            return (
                f"Point to the following items in the image: {objects_str}. "
                "The answer should follow the json format: "
                '[{"point": [y, x], "label": "label1"}, ...]. '
                "The points are in [y, x] format normalized to 0-1000."
            )

    async def predict(
        self,
        images: List[str],
        prompts: List[str],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
    ) -> List[Dict]:
        """Main prediction method for multiple images and prompts."""
        # TODO: Add Gemini segmentation support when available
        # TODO: Add support for Gemini's multimodal capabilities
        assert images, "No images provided"
        assert prompts, "No prompts provided"
        
        results = []
        total_time = 0
        per_image_times = []
        per_prompt_times = []
        
        # Process each image
        for img_idx, image_b64 in enumerate(images):
            image_start_time = time.time()
            image = decode_image(image_b64)
            width, height = image.size
            current_prompt_times = []

            # Process each prompt
            for prompt_idx, prompt in enumerate(prompts):
                prompt_start_time = time.time()
                
                # Format prompt for Gemini
                formatted_prompt = self._format_prompt(prompt, detection)

                # Get predictions from Gemini
                response = self.engine.generate(
                    model=self.model_name,
                    prompt=formatted_prompt,
                    image=image,
                    temperature=0.0,
                    max_tokens=4096,
                )
                
                if self.verbose:
                    console.print(f"\n[bold blue]Gemini Response:[/bold blue]")
                    console.print(f"Prompt: {formatted_prompt}")
                    console.print(f"Response: {response}")
                
                # Parse response
                try:
                    if detection:
                        result = parse_gemini_detection_response(response, (width, height))
                    else:
                        result = parse_gemini_point_response(response, (width, height))
                except Exception as e:
                    console.print(f"[red]Error parsing Gemini response: {str(e)}[/red]")
                    console.print(f"Raw response: {response}")
                    result = {"points": [], "detections": []} if not detection else {"detections": []}

                # Process points and get masks if needed
                formatted_points = []
                masks_data = []
                
                # Process points if points is True
                if points:
                    for point_data in result.get("points", []):
                        assert isinstance(point_data, PointData), "Point data must be PointData object"
                        assert point_data.denormalized_point is not None, "Point must be denormalized"
                        
                        formatted_points.append({
                            "point": point_data.denormalized_point,
                            "label": point_data.label
                        })
                        
                        # Get mask if segmentation is enabled
                        if segmentation:
                            y, x = point_data.denormalized_point
                            mask = ray.get(
                                self.sam_actor.predict.remote(  # type: ignore
                                    image, input=[(x, y)], type="point"
                                )
                            )
                            masks_data.append(mask)

                # Format result
                formatted_result = {
                    "image_index": img_idx,
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "points": formatted_points,
                    "boxes": [] if not detection else [
                        box_data.denormalized_box if box_data.denormalized_box is not None else box_data.box_2d
                        for box_data in result.get("detections", [])
                        if isinstance(box_data, BoxData)
                    ],
                    # NOTE: labels from detection
                    "labels": [] if not detection else [
                        box_data.label
                        for box_data in result.get("detections", [])
                        if isinstance(box_data, BoxData)
                    ],
                    "masks": masks_data,
                    "image_width": width,
                    "image_height": height,
                    "normalization_info": {
                        "width": width,
                        "height": height,
                        "normalized_width": 1000,
                        "normalized_height": 1000,
                    }
                }

                # Save visualization if requested
                if self.save_visualizations:
                    try:
                        vis_image = visualize_results(
                            image,
                            formatted_result,
                            point_radius=10,
                            box_width=2,
                            mask_alpha=0.5,
                        )
                        
                        vis_path = os.path.join(
                            self.visualization_dir,
                            f"vis_img{img_idx}_prompt{prompt_idx}.png"
                        )
                        vis_image.save(vis_path)
                    except Exception as e:
                        console.print(f"[red]Error saving visualization: {str(e)}[/red]")

                results.append(formatted_result)

                # Add timing information
                prompt_time = time.time() - prompt_start_time
                current_prompt_times.append(prompt_time)
                per_prompt_times.append(prompt_time)
            
            image_time = time.time() - image_start_time
            per_image_times.append(image_time)
            total_time += image_time
        
        # Add timing information to results
        timing_info = {
            "total_time": total_time,
            "per_image_times": per_image_times,
            "per_prompt_times": per_prompt_times,
            "avg_time_per_image": total_time / len(images),
            "avg_time_per_prompt": total_time / (len(images) * len(prompts))
        }
        
        if self.verbose:
            console.print("\n[bold blue]Timing Breakdown:[/bold blue]")
            console.print(f"Total time: {total_time:.3f}s", style="cyan")
            console.print(f"Average time per image: {timing_info['avg_time_per_image']:.3f}s", style="cyan")
            console.print(f"Average time per prompt: {timing_info['avg_time_per_prompt']:.3f}s", style="cyan")
            console.print(f"Per image times: {per_image_times}", style="cyan")
            console.print(f"Per prompt times: {per_prompt_times}", style="cyan")
        
        return results

@app.post("/pointing_gemini_sam2_service")
async def predict(
    request: Request,
    verbose: bool = Query(True, description="Enable verbose timing and device information"),
):
    """Handle prediction requests."""
    console.rule("[bold blue]New Request")
    console.print(f"[{Timer.get_current_time()}] Received new request", style="cyan")

    with Timer(enable_print=True, name="FastAPI total handler") as t_total:
        # Time JSON parsing
        with Timer(enable_print=True, name="Request JSON parsing") as t:
            try:
                data = await request.json()
                images = data.get("images", [])
                prompts = data.get("prompts", [])
                points = data.get("points", True)
                segmentation = data.get("segmentation", False)
                detection = data.get("detection", False)

                if not images or not prompts:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Missing required fields: images and/or prompts"},
                    )

                console.print(f"Number of images: {len(images)}", style="cyan")
                console.print(f"Number of prompts: {len(prompts)}", style="cyan")
                console.print(f"Total combinations: {len(images) * len(prompts)}", style="cyan")
                console.print(f"Total input size: {sum(len(img) for img in images) / 1024 / 1024:.2f} MB", style="cyan")
            except Exception as e:
                console.print(f"[red]Error parsing request: {str(e)}[/red]")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request format: {str(e)}"}
                )

        console.print(f"json parsing time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time actual prediction
        with Timer(enable_print=True, name="Service prediction") as t:
            try:
                if service is None:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Service not initialized"}
                    )

                service.verbose = verbose
                result = await service.predict(images, prompts, points, segmentation, detection)
                console.print(f"Prediction completed successfully", style="green")
            except Exception as e:
                console.print(f"[red]Error in prediction: {str(e)}[/red]")
                import traceback
                console.print(traceback.format_exc(), style="red")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Prediction failed: {str(e)}"}
                )

        console.print(f"service prediction time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time response serialization
        with Timer(enable_print=True, name="Response serialization") as t:
            try:
                console.print(f"Response size: {len(str(result)) / 1024 / 1024:.2f} MB", style="cyan")
                return JSONResponse(content=result)
            except Exception as e:
                console.print(f"[red]Error serializing response: {str(e)}[/red]")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to serialize response: {str(e)}"}
                )

        console.print(f"response serialization time: {t.elapsed_time:.3f}s", style="cyan")
    
    console.print(f"[{Timer.get_current_time()}] Request completed", style="cyan")
    console.print("=" * 50 + "\n", style="cyan")

def start_server(
    host: str = "0.0.0.0",
    port: int = 7100,
    use_gpu: bool = True,
    save_visualizations: bool = False,
):
    """Start the FastAPI server."""
    # Create visualization directory if needed
    if save_visualizations:
        os.makedirs("src/models/tmp", exist_ok=True)
        print("Saving visualizations to: src/models/tmp")

    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )

if __name__ == "__main__":
    import typer

    def cli(
        host: str = typer.Option("0.0.0.0", help="Host to bind to"),
        port: int = typer.Option(7100, help="Port to bind to"),
        use_gpu: bool = typer.Option(True, help="Whether to use GPU"),
        save_visualizations: bool = typer.Option(False, help="Whether to save visualizations"),
    ):
        """Start the PointingGeminiSAM2 service"""
        start_server(host, port, use_gpu, save_visualizations)

    typer.run(cli) 