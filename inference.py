import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from safetensors.torch import load_file
from model import Model


def load_model_from_bin(bin_path, device):
    """
    Load model state_dict from .bin file and create model
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Model file not found: {bin_path}")

    print(f"Loading model from: {bin_path}")

    # Load the .bin file
    checkpoint = torch.load(bin_path, map_location=device, weights_only=False)

    # Create model and load state_dict
    model = Model().to(device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume checkpoint is directly the state_dict
        model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully!")
    return model


def load_model_from_safetensors(safetensors_path, device):
    """
    Load model state_dict from safetensors file and create model
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Model file not found: {safetensors_path}")

    print(f"Loading model from: {safetensors_path}")

    # Load the safetensors file
    state_dict = load_file(safetensors_path)

    # Move to device
    state_dict = {key: value.to(device) for key, value in state_dict.items()}

    # Create model and load state_dict
    model = Model().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")
    return model


def inference_on_test_set(model, test_loader, device):
    """
    Run inference on test dataset and calculate accuracy
    """
    print("\nRunning inference on test set...")

    all_correct_num = 0
    all_sample_num = 0

    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)

            # Forward pass
            predict_y = model(test_x.float())
            predict_y = torch.argmax(predict_y, dim=-1)

            # Calculate accuracy
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to("cpu").numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} batches...")

    accuracy = all_correct_num / all_sample_num
    print(f"\nTest Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    print(f"Using device: {device}")

    # Model file path - use safetensors format
    safetensors_model_path = "./models/model-mnist.safetensors"

    # Load model from safetensors
    model = load_model_from_safetensors(safetensors_model_path, device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = mnist.MNIST(root="./test", train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Run inference on test set
    # accuracy = inference_on_test_set(model, test_loader, device)

    # Test on a few single images
    print("\n" + "=" * 50)
    print("Testing on individual images:")
    print("=" * 50)

    for i in range(5):
        test_image, test_label = test_dataset[i]
        test_image = test_image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            image_tensor = test_image.to(device)
            predict_y = model(image_tensor.float())
            predict_class = torch.argmax(predict_y, dim=-1)

        print(
            f"Image {i}: True label = {test_label}, Predicted = {predict_class}, "
            f"Match = {predict_class == test_label}"
        )

    print("\nInference completed!")
