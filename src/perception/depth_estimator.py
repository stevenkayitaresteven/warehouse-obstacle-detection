"""
Depth Anything V2 relative-depth estimation module.
"""

import torch
import torch.nn.functional as F

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


class RelativeDepthEstimator:
    """
    Generate dense relative-depth maps using Depth Anything V2.

    IMPORTANT:
    The output is relative depth, not calibrated metric distance.
    """

    def __init__(
        self,
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        device=None,
    ):
        self.model_id = model_id

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def estimate(self, image):
        """
        Estimate a relative-depth map.

        Parameters
        ----------
        image:
            PIL image or path to an image.

        Returns
        -------
        numpy.ndarray:
            Float32 relative-depth map with shape (height, width).
        """

        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        original_width, original_height = image.size

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return depth_map.cpu().numpy().astype("float32")
