import os
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image

# -----------------------------
# Settings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_samples = 1000
image_size = 28

os.makedirs("samples/unconditional", exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=1
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000
).to(device)

model.load_state_dict(torch.load("mnist_unconditional_unet.pt", map_location=device))
diffusion.load_state_dict(torch.load("mnist_unconditional_diffusion.pt", map_location=device))

model.eval()
diffusion.eval()

# -----------------------------
# Sample
# -----------------------------
with torch.no_grad():
    samples = diffusion.sample(batch_size=num_samples)

# -----------------------------
# Save
# -----------------------------
for i, img in enumerate(samples):
    save_image(
        (img + 1) / 2,
        f"samples/unconditional/{i:04d}.png"
    )

print("Saved unconditional samples to samples/unconditional/")
