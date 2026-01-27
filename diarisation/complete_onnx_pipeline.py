# complete_onnx_pipeline.py
import os
import sys
from pathlib import Path

# Step 1: Export model to ONNX (run once)
def setup_model(hf_token: str, model_path: str = "segmentation_model.onnx"):
    """Export model if it doesn't exist."""
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return model_path
    
    print("Exporting model to ONNX...")
    import torch
    from pyannote.audio import Model
    
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        token=hf_token
    ).eval()
    
    dummy_input = torch.zeros(2, 1, 160000)
    
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
            "logits": {0: "batch_size", 1: "num_frames"},
        },
        opset_version=14,
    )
    
    print(f"✓ Model exported to {model_path}")
    return model_path

if __name__ == "__main__":
    # Get HF token from environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)
    AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    
    try:
        # Export model (once)
        model_path = setup_model(HF_TOKEN)
        
        # Import diarization class
        from onnx_diarization import ONNXSpeakerDiarization
        
        # Run diarization
        diarizer = ONNXSpeakerDiarization(model_path)
        segments = diarizer.diarize(AUDIO_FILE)
        
        # Display results
        print("\n📊 Results:")
        for start, end, speaker in segments:
            print(f"  SPEAKER_{speaker:02d}: {start:.2f}s → {end:.2f}s")
        
        # Save RTTM
        audio_name = Path(AUDIO_FILE).stem
        diarizer.save_rttm(segments, f"{audio_name}.rttm", audio_name)
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)