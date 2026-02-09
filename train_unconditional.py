import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm import tqdm

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # [0,1] -> [-1,1]
])

# dataset
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=1
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=28,
    timesteps=1000
).to(device)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

epochs = 10

for epoch in range(epochs):
    pbar = tqdm(dataloader)
    for images, _ in pbar:
        images = images.to(device)
        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Save models after training
torch.save(model.state_dict(), "mnist_unconditional_unet.pt")
torch.save(diffusion.state_dict(), "mnist_unconditional_diffusion.pt")
print("Models saved successfully!")
