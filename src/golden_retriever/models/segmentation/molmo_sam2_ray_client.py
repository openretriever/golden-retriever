import asyncio
import base64
import io
import os
from typing import Dict, Union

import ray
import typer
from PIL import Image
from ray import serve


def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string"""
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
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


class MolmoSAM2Client:
    def __init__(self, host: str = "localhost", port: int = 8100):
        """Initialize client with host and port"""
        self.endpoint = f"http://{host}:{port}"
        self.handle = None

    def connect(self):
        """Connect to the Ray Serve deployment"""
        try:
            print("Initializing Ray...")
            if not ray.is_initialized():
                ray.init(address="auto")  # Try to connect to existing Ray cluster
            print("Ray initialized. Getting deployment handle...")

            # Specify the application name when getting the deployment handle
            self.handle = serve.get_deployment_handle(
                "molmo_sam2_service",
                app_name="default",  # Use the default application name or specify your own
            )
            print("Deployment handle obtained.")
            return self
        except Exception as e:
            print(f"Failed to connect to Ray Serve deployment: {e}")
            print(
                "Make sure the server is running with: python -m src.models.segmentation.molmo_sam2_server"
            )
            raise

    async def predict_async(self, image: Union[str, Image.Image], prompt: str) -> Dict:
        """
        Async prediction using Ray Serve handle

        Args:
            image: Path to image or PIL Image
            prompt: Text prompt for point selection
        """
        if self.handle is None:
            self.connect()

        # Encode image if needed
        image_b64 = encode_image(image)

        # Make request using handle
        response = await self.handle.remote({"image": image_b64, "prompt": prompt})

        # Convert segmentation image back to PIL Image
        if isinstance(response, dict) and "segmentation_image" in response:
            response["segmentation_image"] = decode_image(
                response["segmentation_image"]
            )

        return response

    def predict(self, image: Union[str, Image.Image], prompt: str) -> Dict:
        """
        Synchronous prediction method
        """
        return asyncio.run(self.predict_async(image, prompt))


app = typer.Typer()


@app.command()
def predict(
    image_path: str = typer.Argument(..., help="Path to the input image"),
    prompt: str = typer.Argument(..., help="Text prompt for point selection"),
    host: str = typer.Option("localhost", help="Host of the MolmoSAM2 service"),
    port: int = typer.Option(8100, help="Port of the MolmoSAM2 service"),
    output: str = typer.Option("segmentation_result.png", help="Output file path"),
):
    """Run a prediction using the MolmoSAM2 service"""
    # Join prompt words into a single string
    # prompt_text = " ".join(prompt)
    prompt_text = prompt

    client = MolmoSAM2Client(host=host, port=port)
    try:
        print(f"Running prediction with prompt: {prompt_text}")

        result = client.predict(image_path, prompt_text)
        tmp_dir = os.path.join("src", "models", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        output_path = os.path.join(tmp_dir, output)
        result["segmentation_image"].save(output_path)
        print(f"Saved segmentation result to: {output_path}")
        print("Points:", result["points"])
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print(
            "1. Start the server first: python -m src.models.segmentation.molmo_sam2_server"
        )
        print("2. Wait a few seconds for the server to initialize")
        print("3. Try the request again")


if __name__ == "__main__":
    app()
