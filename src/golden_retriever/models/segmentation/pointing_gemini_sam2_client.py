"""Client for Gemini pointing capability with optional SAM2 segmentation."""

import asyncio
import base64
import io
import os
import time
from typing import Dict, List, Union, Sequence, cast, Any

import httpx
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

from retriever.models.api_models.utils.gemini_parsing import visualize_results
from retriever.models.common_utils import Timer

# Initialize console globally
console = Console()


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


class PointingGeminiSAM2Client:
    """HTTP client for the Pointing Gemini SAM2 service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7100,
        timeout: int = 300,
    ):
        """Initialize the client."""
        # TODO: Add Gemini segmentation support when available
        # TODO: Add support for Gemini's multimodal capabilities
        self.host = host
        self.endpoint = f"http://{host}:{port}/pointing_gemini_sam2_service"

    async def predict_async(
        self,
        images: Union[str, Image.Image, Sequence[Union[str, Image.Image]]],
        prompts: Union[str, Sequence[str]],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
    ) -> Dict[str, Any]:
        """Async prediction with support for multiple images and prompts."""
        timings = {}
        console.rule("[bold blue]Starting PointingGeminiSAM2 Client Request")
        console.print(f"Endpoint: {self.endpoint}", style="cyan")

        # Convert single inputs to lists
        if isinstance(images, (str, Image.Image)):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process images
        processed_images = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing images...", total=len(images))
            
            with Timer(enable_print=False) as t:
                for i, img in enumerate(images):
                    if isinstance(img, str):
                        progress.update(task, description=f"Loading image {i+1}: {os.path.basename(img)}")
                        img_pil = Image.open(img)
                    else:
                        progress.update(task, description=f"Processing image {i+1}")
                        img_pil = img
                    
                    img_b64 = encode_image(img_pil)
                    processed_images.append(img_b64)
                    progress.advance(task)
            timings["preprocessing"] = t.elapsed_time

        # Make server request
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Waiting for server response...")
            
            with Timer(enable_print=False) as t:
                try:
                    async with httpx.AsyncClient(timeout=600.0) as client:  # Increased timeout to 10 minutes
                        response = await client.post(
                            self.endpoint,
                            json={
                                "images": processed_images,
                                "prompts": prompts,
                                "points": points,
                                "segmentation": segmentation,
                                "detection": detection,
                            },
                            timeout=600.0  # Increased timeout to 10 minutes
                        )
                        
                        if response.status_code != 200:
                            error_msg = f"Server returned status code {response.status_code}: {response.text}"
                            console.print(f"[red]Error: {error_msg}[/red]")
                            raise httpx.HTTPError(error_msg)
                        
                        progress.update(task, completed=True)
                        result = response.json()
                        console.print("[green]Server request completed successfully[/green]")
                except httpx.TimeoutException as e:
                    console.print(f"[red]Request timed out after 10 minutes. Try reducing the number of images or prompts.[/red]")
                    raise
                except httpx.HTTPError as e:
                    console.print(f"[red]HTTP error occurred: {str(e)}[/red]")
                    raise
                except Exception as e:
                    console.print(f"[red]Unexpected error occurred: {str(e)}[/red]")
                    raise
            timings["server_request"] = t.elapsed_time

        return {"results": result, "timings": timings}

    def predict(
        self,
        images: Union[str, Image.Image, Sequence[Union[str, Image.Image]]],
        prompts: Union[str, Sequence[str]],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
        save_visualization: bool = True,
        visualization_dir: str = "output",
        point_radius: int = 10,
        box_width: int = 2,
        mask_alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for predict_async."""
        try:
            with Timer(enable_print=False) as t:
                result = asyncio.run(self.predict_async(
                    images, prompts, points, segmentation, detection
                ))
            result["timings"]["total"] = t.elapsed_time

            # Handle visualization if requested
            if save_visualization:
                # Create visualization directory if needed
                os.makedirs(visualization_dir, exist_ok=True)

                # Process each result
                for res in result["results"]:
                    img_idx = res["image_index"]
                    prompt_idx = res["prompt_index"]
                    
                    # Load and visualize image
                    current_image: Image.Image
                    if isinstance(images, (str, Image.Image)):
                        current_image = images if isinstance(images, Image.Image) else Image.open(images)
                    else:
                        current_image = images[img_idx] if isinstance(images[img_idx], Image.Image) else Image.open(images[img_idx])
                    
                    # Make a copy of the image for this visualization
                    vis_image = current_image.copy()
                    
                    # Create visualization
                    vis_image = visualize_results(
                        vis_image,
                        res,
                        point_radius=point_radius,
                        box_width=box_width,
                        mask_alpha=mask_alpha,
                    )
                    
                    # Save visualization
                    vis_path = os.path.join(
                        visualization_dir,
                        f"vis_img{img_idx}_prompt{prompt_idx}.png"
                    )
                    vis_image.save(vis_path)
                    console.print(f"[green]Saved visualization to:[/green] {vis_path}")

            return result
        except Exception as e:
            console.print(f"[red]Error during prediction: {str(e)}[/red]")
            raise


app = typer.Typer()


@app.command()
def predict(
    image: List[str] = typer.Option(
        None,
        "--image", "-i",
        help="Image path(s). Can be specified multiple times for multiple images.",
        callback=lambda x: x or [],
    ),
    prompt: List[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="Text prompt(s). Can be specified multiple times for multiple prompts.",
        callback=lambda x: x or [],
    ),
    host: str = typer.Option("localhost", help="Host of the PointingGeminiSAM2 service"),
    port: int = typer.Option(7100, help="Port of the PointingGeminiSAM2 service"),
    points: bool = typer.Option(True, help="Whether to return point coordinates"),
    segmentation: bool = typer.Option(False, help="Whether to perform segmentation"),
    detection: bool = typer.Option(False, help="Whether to perform detection instead of pointing"),
    save_visualization: bool = typer.Option(True, help="Whether to save visualizations"),
    visualization_dir: str = typer.Option("output", help="Directory to save visualizations"),
    point_radius: int = typer.Option(10, help="Radius of point markers in pixels"),
    box_width: int = typer.Option(2, help="Width of box lines in pixels"),
    mask_alpha: float = typer.Option(0.5, help="Opacity of masks (0-1)"),
):
    """Run predictions using the PointingGeminiSAM2 service."""
    try:
        # Clean up inputs
        prompts = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in prompt]
        image_paths = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in image]

        if not image_paths or not prompts:
            raise typer.BadParameter("Must provide at least one image (-i) and one prompt (-p)")

        client = PointingGeminiSAM2Client(host=host, port=port)
        result = client.predict(image_paths, prompts, points=points, segmentation=segmentation, detection=detection)

        # Print results
        console.rule("[bold blue]Results")
        for res in result["results"]:
            img_idx = res["image_index"]
            prompt_idx = res["prompt_index"]
            prompt = res["prompt"]
            
            console.print(f"\n[yellow]Image {img_idx + 1}, Prompt {prompt_idx + 1}: '{prompt}'[/yellow]")
            
            # Print points if available
            if res.get("points"):
                if len(res["points"]) == 0:
                    console.print("[red]No points found in the image[/red]")
                else:
                    console.print(f"Points coordinates: {res['points']}", style="cyan")
            
            # Print detections if available
            if res.get("boxes") and res.get("labels"):
                if len(res["boxes"]) == 0:
                    console.print("[red]No objects detected[/red]")
                else:
                    console.print("\nDetected objects:", style="cyan")
                    for i, (box, label) in enumerate(zip(res["boxes"], res["labels"]), 1):
                        console.print(f"Object {i}: {box} ({label})", style="cyan")
            
            # Print masks if available
            if res.get("masks"):
                if len(res["masks"]) == 0:
                    console.print("[red]No masks generated[/red]")
                else:
                    console.print(f"Generated {len(res['masks'])} masks", style="cyan")

            # Save visualization if requested
            if save_visualization:
                # Create visualization directory if needed
                os.makedirs(visualization_dir, exist_ok=True)
                
                # Load and visualize image
                current_image: Image.Image
                if isinstance(image_paths, (str, Image.Image)):
                    current_image = image_paths if isinstance(image_paths, Image.Image) else Image.open(image_paths)
                else:
                    current_image = image_paths[img_idx] if isinstance(image_paths[img_idx], Image.Image) else Image.open(image_paths[img_idx])
                
                # Create visualization
                vis_image = current_image.copy()
                
                # Create visualization
                vis_image = visualize_results(
                    vis_image,
                    res,
                    point_radius=point_radius,
                    box_width=box_width,
                    mask_alpha=mask_alpha,
                )
                
                # Save visualization
                vis_path = os.path.join(
                    visualization_dir,
                    f"vis_img{img_idx}_prompt{prompt_idx}.png"
                )
                vis_image.save(vis_path)
                console.print(f"[green]Saved visualization to:[/green] {vis_path}")

        # Print timing
        console.rule("[bold blue]Timing")
        console.print(f"Preprocessing: {result['timings']['preprocessing']:.3f}s", style="cyan")
        console.print(f"Server request: {result['timings']['server_request']:.3f}s", style="cyan")
        console.print(f"Total time: {result['timings']['total']:.3f}s", style="green")

    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print("\n[yellow]Make sure to:[/yellow]")
        console.print("1. Start the server first: [cyan]python -m src.models.segmentation.pointing_gemini_sam2[/cyan]")
        console.print("2. Wait a few seconds for the server to initialize")
        console.print("3. Try the request again")


if __name__ == "__main__":
    app() 