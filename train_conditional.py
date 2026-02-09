import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 10
num_labels = 10  # MNIST digits

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # [0,1] -> [-1,1]
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Model: 2-channel input (image + label)
# -----------------------------
# Channel 1: image, Channel 2: label embedding as image
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=2  # <-- extra channel for label
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=28,
    timesteps=1000
).to(device)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(epochs):
    pbar = tqdm(dataloader)
    for images, labels in pbar:
        images = images.to(device)  # [B,1,28,28]

        # -----------------------------
        # Convert labels to 2nd channel
        # -----------------------------
        labels_channel = labels[:, None, None, None].float()       # [B,1,1,1]
        labels_channel = labels_channel / (num_labels - 1) * 2 - 1 # normalize to [-1,1]
        labels_channel = labels_channel.expand(-1, 1, 28, 28)     # [B,1,28,28]

        # Combine image + label channel
        x_cond = torch.cat([images, labels_channel], dim=1)       # [B,2,28,28]

        # Forward pass
        loss = diffusion(x_cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -----------------------------
# Save models
# -----------------------------
torch.save(model.state_dict(), "mnist_conditional_unet.pt")
torch.save(diffusion.state_dict(), "mnist_conditional_diffusion.pt")
print("Conditional models saved successfully!")
