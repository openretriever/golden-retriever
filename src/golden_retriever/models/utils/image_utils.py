"""Shared utilities for image and result handling."""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from datetime import datetime
from PIL import Image
import pillow_heif
import pyheif
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

# Initialize console for utilities
console = Console()


def load_images(image_dir: str) -> Tuple[List[Image.Image], List[str]]:
    """Load all images from the specified directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Tuple of (list of PIL Images, list of image paths)
    """
    images = []
    image_paths = []
    
    # Get sorted list of files
    file_paths = (
        sorted(Path(image_dir).glob("*.jpg")) + 
        sorted(Path(image_dir).glob("*.png")) +
        sorted(Path(image_dir).glob("*.heic")) +
        sorted(Path(image_dir).glob("*.HEIC"))  # Add uppercase extension
    )
    
    if not file_paths:
        console.print("[yellow]Warning: No image files found in directory[/yellow]")
        return images, image_paths
    
    for file_path in track(file_paths, description="Loading images"):
        try:
            if file_path.suffix.lower() in ['.heic']:  # Check lowercase extension
                # Try loading with pyheif first (since it's working for your files)
                try:
                    heif_file = pyheif.read(file_path)
                    img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                    )
                except Exception as e:
                    console.print(f"[yellow]pyheif failed, trying pillow-heif: {str(e)}[/yellow]")
                    # If pyheif fails, try pillow-heif
                    heif_file = pillow_heif.read_heif(file_path)
                    if heif_file:
                        img = Image.frombytes(
                            heif_file.mode,
                            heif_file.size,
                            heif_file.data,
                            "raw",
                        )
                    else:
                        raise ValueError("Could not read HEIC file with either library")
            else:
                # Handle regular images
                img = Image.open(file_path)
            
            # Convert to RGB mode if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            images.append(img)
            image_paths.append(str(file_path))
            console.print(f"[green]✓[/green] Loaded {file_path.name}")
            
        except Exception as e:
            console.print(Panel(
                f"[red]Error loading image:[/red]\n"
                f"File: {file_path.name}\n"
                f"Error: {str(e)}",
                title="Loading Error",
                border_style="red"
            ))
            continue
            
    if not images:
        console.print("[red]No images were successfully loaded[/red]")
    else:
        console.print(f"[green]Successfully loaded {len(images)} images[/green]")
        
    return images, image_paths


def load_queries(query_file: str) -> List[str]:
    """Load queries from a text file.
    
    Args:
        query_file: Path to file containing queries
        
    Returns:
        List of queries
    """
    try:
        with open(query_file, 'r') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
        console.print(f"[green]Successfully loaded {len(queries)} queries[/green]")
        return queries
    except Exception as e:
        console.print(f"[red]Error loading queries:[/red] {str(e)}")
        return []


def save_results(results: List[Dict], output_file: str) -> str:
    """Save results to YAML format.
    
    Args:
        results: List of result dictionaries
        output_file: Base name for output file
        
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_filename = f"{output_file}_{timestamp}.yaml"
    
    try:
        with open(yaml_filename, 'w') as f:
            yaml.dump(results, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        console.print(f"[green]Results saved to[/green] {yaml_filename}")
        return yaml_filename
    except Exception as e:
        console.print(f"[red]Error saving YAML:[/red] {str(e)}")
        return ""


def ensure_directories(*dirs: str) -> None:
    """Ensure that the specified directories exist.
    
    Args:
        *dirs: Directory paths to create if they don't exist
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True) 