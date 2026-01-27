# export_to_onnx.py
import os
import torch
from pyannote.audio import Model

def export_segmentation_model(hf_token: str, output_path: str = "segmentation_model.onnx"):
    """Export pyannote segmentation model to ONNX format."""
    
    print("Loading segmentation model...")
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        token=hf_token  # Updated parameter name
    ).eval()
    
    print(f"Exporting model to {output_path}...")
    
    # Create dummy input (batch_size=2, channels=1, samples=160000)
    # 160000 samples = 10 seconds at 16kHz
    dummy_input = torch.zeros(2, 1, 160000)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
            "logits": {0: "batch_size", 1: "num_frames"},
        },
        opset_version=14,
    )
    
    print(f"✓ Model exported successfully to {output_path}")
    return output_path

if __name__ == "__main__":
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set")
        exit(1)
    export_segmentation_model(HF_TOKEN)