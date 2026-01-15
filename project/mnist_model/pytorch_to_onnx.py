#function to convert pytorch model to onnx
import torch
from mnist_model import NeuralNetwork
import os
from pathlib import Path    
import sys 

#main function
if __name__ == "__main__":
    # Load the trained PyTorch model
    model = NeuralNetwork()
    model.load_state_dict(torch.load("mnist_model/results/mnist_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Create output directory if it doesn't exist
    OUT_DIR = Path("mnist_model")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dummy input for the model (batch size 1, 1 channel, 28x28 image)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export the model to ONNX format
    onnx_path = OUT_DIR / "mnist_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)
    
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

