"""Standalone version of Gemini pointing capability with optional SAM2 segmentation."""

import os
import base64
import io
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import time

import ray
import torch
from PIL import Image, ImageDraw
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
    denormalize_point,
    denormalize_box,
)
from retriever.models.api_models.utils.google_utils import GeminiClient, GEMINI_MODEL_NAME
from retriever.models.common_utils import Timer

# Initialize console globally
console = Console()
install(show_locals=True)


def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string."""
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    else:
        raise ValueError("Image must be either a file path or PIL Image")


def decode_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def draw_points(
    image: Image.Image,
    points_data: List[Dict],
    point_radius: int = 10,
    text_offset: int = 15,
) -> Image.Image:
    """Draw points and labels on the image with enhanced visibility."""
    draw = ImageDraw.Draw(image)
    
    for point_data in points_data:
        point = point_data["point"]
        label = str(point_data["label"])
        
        # Draw point with larger radius and white outline
        draw.ellipse(
            [
                point[1] - point_radius,
                point[0] - point_radius,
                point[1] + point_radius,
                point[0] + point_radius,
            ],
            fill="red",
            outline="white",
            width=2,
        )
        
        # Draw label with stroke for better visibility
        draw.text(
            (point[1] + text_offset, point[0] - text_offset),
            label,
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )
    
    return image


def draw_boxes(
    image: Image.Image,
    boxes_data: List[List[float]],
    labels: List[str],
    box_width: int = 2,
    text_offset: int = 10,
) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    
    for box, label in zip(boxes_data, labels):
        # Draw box
        draw.rectangle(
            (box[0], box[1], box[2], box[3]),
            outline="red",
            width=box_width,
        )
        
        # Draw label
        draw.text(
            (box[0] + text_offset, box[1] - text_offset),
            label,
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )
    
    return image


def draw_masks(
    image: Image.Image,
    masks_data: List[Image.Image],
    alpha: float = 0.5,
) -> Image.Image:
    """Draw segmentation masks on the image."""
    if not masks_data:
        return image
    
    # Convert image to RGBA if needed
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a new image for the overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Draw each mask
    for mask in masks_data:
        # Convert mask to RGBA
        mask_rgba = mask.convert('RGBA')
        # Create a colored version of the mask
        colored_mask = Image.new('RGBA', mask.size, (255, 0, 0, int(255 * alpha)))
        # Apply the mask
        colored_mask.putalpha(mask_rgba.split()[3])
        # Composite onto overlay
        overlay = Image.alpha_composite(overlay, colored_mask)
    
    # Composite overlay onto original image
    return Image.alpha_composite(image, overlay)


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
    ):
        """Initialize the service."""
        self.verbose = verbose
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir

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
                "Output a json list where each entry contains the 2D bounding box in \"box_2d\" and a text label in \"label\"."
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
        results = []
        total_time = 0
        gemini_time = 0
        sam_time = 0
        visualization_time = 0
        
        # Process each image
        for img_idx, image_b64 in enumerate(images):
            start_time = time.time()
            
            # Decode image
            image = decode_image(image_b64)
            width, height = image.size

            # Process each prompt
            for prompt_idx, prompt in enumerate(prompts):
                # Format prompt for Gemini
                formatted_prompt = self._format_prompt(prompt, detection)

                # Get predictions from Gemini
                gemini_start = time.time()
                response = self.engine.generate(
                    model=self.model_name,
                    prompt=formatted_prompt,
                    image=image,
                    temperature=0.0,
                    max_tokens=4096,
                )
                gemini_time += time.time() - gemini_start
                
                # Parse response
                if detection:
                    result = parse_gemini_detection_response(response, (width, height))
                else:
                    result = parse_gemini_point_response(response, (width, height))

                # Get masks from SAM2 if requested
                masks_data = []
                if segmentation:
                    sam_start = time.time()
                    if detection:
                        for box_data in result["detections"]:
                            box = cast(List[float], box_data["box_2d"])
                            # Convert normalized box to image coordinates
                            x1, y1, x2, y2 = denormalize_box(box, (width, height))
                            mask = ray.get(
                                self.sam_actor.predict.remote(  # type: ignore
                                    image, input=[(x1, y1, x2, y2)], type="box"
                                )
                            )
                            masks_data.append(mask)
                    else:
                        for point_data in result["points"]:
                            point = cast(List[float], point_data["point"])
                            # Convert normalized point to image coordinates
                            y, x = denormalize_point(point, (width, height))
                            mask = ray.get(
                                self.sam_actor.predict.remote(  # type: ignore
                                    image, input=[(x, y)], type="point"
                                )
                            )
                            masks_data.append(mask)
                    sam_time += time.time() - sam_start

                # Format result to match Molmo's format
                formatted_result = {
                    "image_index": img_idx,
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "points": result.get("points", []),
                    "boxes": [
                        [float(x) for x in box["box_2d"]]
                        for box in result.get("detections", [])
                    ] if result.get("detections") else None,
                    "masks": masks_data if segmentation else None,
                    "image_width": width,
                    "image_height": height,
                    "image": image_b64
                }
                results.append(formatted_result)

                # Save visualizations if requested
                if self.save_visualizations:
                    vis_start = time.time()
                    # Create visualization directory if needed
                    os.makedirs(self.visualization_dir, exist_ok=True)
                    
                    # Create base visualization
                    vis_image = image.copy()
                    
                    # Draw points if available
                    if formatted_result["points"]:
                        vis_image = draw_points(vis_image, formatted_result["points"])
                    
                    # Draw boxes if available
                    if formatted_result["boxes"]:
                        vis_image = draw_boxes(
                            vis_image,
                            formatted_result["boxes"],
                            [box["label"] for box in result.get("detections", [])]
                        )
                    
                    # Draw masks if available
                    if formatted_result["masks"]:
                        vis_image = draw_masks(vis_image, formatted_result["masks"])
                    
                    # Save visualization
                    vis_path = os.path.join(
                        self.visualization_dir,
                        f"vis_img{img_idx}_prompt{prompt_idx}.png"
                    )
                    vis_image.save(vis_path)
                    visualization_time += time.time() - vis_start

            total_time += time.time() - start_time

        # Add timing information to results
        timing_info = {
            "total_time": total_time,
            "gemini_time": gemini_time,
            "sam_time": sam_time,
            "visualization_time": visualization_time,
            "avg_time_per_image": total_time / len(images) if images else 0,
            "avg_time_per_prompt": total_time / (len(images) * len(prompts)) if images and prompts else 0
        }
        
        if self.verbose:
            console.print("\n[bold blue]Timing Breakdown:[/bold blue]")
            console.print(f"Total time: {total_time:.3f}s", style="cyan")
            console.print(f"Gemini time: {gemini_time:.3f}s", style="cyan")
            console.print(f"SAM time: {sam_time:.3f}s", style="cyan")
            console.print(f"Visualization time: {visualization_time:.3f}s", style="cyan")
            console.print(f"Average time per image: {timing_info['avg_time_per_image']:.3f}s", style="cyan")
            console.print(f"Average time per prompt: {timing_info['avg_time_per_prompt']:.3f}s", style="cyan")

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