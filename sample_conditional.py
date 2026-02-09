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
num_labels = 10

os.makedirs("samples/conditional", exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=2
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000
).to(device)

model.load_state_dict(torch.load("mnist_conditional_unet.pt", map_location=device))
diffusion.load_state_dict(torch.load("mnist_conditional_diffusion.pt", map_location=device))

model.eval()
diffusion.eval()

# -----------------------------
# Choose labels to condition on
# -----------------------------
labels = torch.randint(0, num_labels, (num_samples,), device=device)

labels_channel = labels[:, None, None, None].float()
labels_channel = labels_channel / (num_labels - 1) * 2 - 1
labels_channel = labels_channel.expand(-1, 1, image_size, image_size)

# -----------------------------
# Sample
# -----------------------------
with torch.no_grad():
    samples = diffusion.sample(
        batch_size=num_samples,
        cond=labels_channel   # 👈 THIS is the key
    )

# -----------------------------
# Save only image channel
# -----------------------------
for i, img in enumerate(samples[:, 0:1]):  # take image channel only
    save_image(
        (img + 1) / 2,
        f"samples/conditional/{i:04d}.png"
    )

print("Saved conditional samples to samples/conditional/")
