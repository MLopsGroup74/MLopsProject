
from src.assignment.model import ConvolutionalNetwork


def test_convolutional_network() -> None:
    """Test that ConvolutionalNetwork forward pass works and outputs correct shape."""
    num_classes = 150
    batch_size = 4
    img_size = 224
    import torch
    x = torch.randn(batch_size, 3, img_size, img_size)

    model = ConvolutionalNetwork(num_classes=num_classes)

    out = model(x)

    assert out.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"

    assert isinstance(out, torch.Tensor), "Output is not a torch.Tensor"

    print(" ConvolutionalNetwork forward pass test passed.")
