import torch
from src.assignment.model import ConvolutionalNetwork

def test_convolutional_network():
    """Test that ConvolutionalNetwork forward pass works and outputs correct shape."""
    num_classes = 150  # adjust to your dataset
    batch_size = 4
    img_size = 224

    # Create a dummy batch of images (B, C, H, W)
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize model
    model = ConvolutionalNetwork(num_classes=num_classes)

    # Forward pass
    out = model(x)

    # Check output shape
    assert out.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"

    # Check output type
    assert isinstance(out, torch.Tensor), "Output is not a torch.Tensor"

    print(" ConvolutionalNetwork forward pass test passed.")

