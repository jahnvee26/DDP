from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

# Dynamically select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load once at import time!
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

def run_vla_action(image, prompt):
    """
    Run VLA model for a given image + prompt.
    Returns action dict.
    """
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return action

if __name__ == "__main__":
    from PIL import Image

    # === 1. Load a test image from disk ===
    test_image = Image.open("/home/janvi/Downloads/image.jpg")

    # === 2. Test prompt ===
    prompt = "In: What action should the robot take to pick up the red cube?\nOut:"

    # === 3. Run VLA ===
    action = run_vla_action(test_image, prompt)
    print(f"[TEST] VLA predicted action: {action}")
