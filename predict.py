# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'
CONTROLNET_CACHE = "FLUX.1-dev-ControlNet-Union-Pro"
CONTROLNET_MODEL_UNION = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CONTROLNET_URL = "https://weights.replicate.delivery/default/shakker-labs/FLUX.1-dev-ControlNet-Union-Pro/model.tar"

CONTROL_TYPES = ["canny", "tile", "depth", "blur", "pose", "gray", "low-quality", "none"]

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(CONTROLNET_CACHE):
            download_weights(CONTROLNET_URL, CONTROLNET_CACHE)
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")

        controlnet_union = FluxControlNetModel.from_pretrained(
            CONTROLNET_CACHE,
            torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])
        self.pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="A bohemian-style female travel blogger with sun-kissed skin and messy beach waves"),
        guidance_scale: float = Input(description="Guidance scale", default=3.5, ge=0, le=5),
        steps: int = Input(description="Number of steps", default=28, ge=1, le=50),
        control_type: str = Input(description="Control type", default="canny", choices=CONTROL_TYPES),
        control_strength: float = Input(description="ControlNet strength, depth works best at 0.2, canny works best at 0.4. Recommended range is 0.3-0.8", default=0.2, ge=0, le=1),
        control_image: Path = Input(description="Control image", default=""),
        control_type_2: str = Input(description="Control type", default="none", choices=CONTROL_TYPES),
        control_strength_2: float = Input(description="ControlNet strength, depth works best at 0.2, canny works best at 0.4. Recommended range is 0.3-0.8", default=0.2, ge=0, le=1),
        control_image_2: Path = Input(description="Control image", default=""),
        seed: int = Input(description="Set a seed for reproducibility. Random by default.", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # set control mode to one of 7 supported control types
        control_mode = CONTROL_TYPES.index(control_type)
        control_mode_2 = CONTROL_TYPES.index(control_type_2)
        control_image = Image.open(control_image)
        control_image_2 = Image.open(control_image_2)
        width, height = control_image.size
        #resize image to be divisible by 8
        control_image = control_image.resize((width // 8 * 8, height // 8 * 8))
        control_image_2 = control_image_2.resize((width // 8 * 8, height // 8 * 8))
        width, height = control_image.size

        control_images = [control_image]
        control_modes = [control_mode]
        control_strengths = [control_strength]

        if control_image_2 != "none":
            control_images.append(control_image_2)
            control_modes.append(control_mode_2)
            control_strengths.append(control_strength_2)

        image = self.pipe(
            prompt,
            control_image=control_images,
            control_mode=control_modes,
            width=width,
            height=height,
            controlnet_conditioning_scale=control_strengths,
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        output_path = f"/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
