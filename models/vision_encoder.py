"""CNN encoder for RGB partial observations."""
import torch
from torch import nn

class VisionEncoder(nn.Module):
    """Convert an RGB partial observation into a fixed-size embedding."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            # This makes the encoder work with different RGB image sizes.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),      
        )

        self.projection = nn.Sequential(
            nn.Linear(64 * 4 * 4, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor):
        # Encode images.
        """
        Args:
            images:
                Tensor with shape ``(batch, 3, height, width)``.

                Pixel values may be uint8 values in [0, 255] or floating-point
                values in [0, 255]. They are converted to float32 and scaled
                to [0, 1] here.

        Returns:
            Tensor with shape ``(batch, embedding_dim)``.
        """
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape (batch, channels, height, width), "
                f"but received {tuple(images.shape)}."
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, but received {images.shape[1]}."
            )

        images = images.to(dtype=torch.float32) / 255.0
        features = self.cnn(images)
        return self.projection(features)
    