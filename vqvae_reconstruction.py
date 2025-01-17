import argparse
import os, sys
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from vqvae import FlatVQVAE
import distributed as dist

def reconstruct_100Class(model, loader, device, save_path):
    # output_dir = "reconstructed_images"
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        total_recon_loss = 0
        criterion = nn.MSELoss()  # Define the reconstruction loss
        num_samples = 0

        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.float().to(device)
            outputs, _, _, _ = model(images)  # Get the model outputs

            # Calculate reconstruction loss for the current batch
            recon_loss = criterion(outputs, images)
            total_recon_loss += recon_loss.item() * images.size(0)  # Accumulate weighted loss
            num_samples += images.size(0)  # Update total sample count

            # Save reconstructed images
            for idx, (img, out) in enumerate(zip(images, outputs)):
                save_file = os.path.join(save_path, f"reconstructed_{i * loader.batch_size + idx + 1:05d}.png")
                utils.save_image(
                    torch.cat([out.unsqueeze(0)], 0),
                    save_file,
                    nrow=2,
                    normalize=True,
                    range=(-1, 1),
                )

        # Compute and print the average reconstruction loss
        avg_recon_loss = total_recon_loss / num_samples
        print(f"Average Reconstruction Loss: {avg_recon_loss:.5f}")


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    torch.cuda.set_device(1)  # Use GPU 1 (if desired)

    # 2. Clear any cached memory on the selected GPU
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transform to normalize data and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor (0-1 range)
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])

    # Load MNIST dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Create DataLoaders
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

    # Load the model
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae.eval()

    # Perform reconstruction
    reconstruct_100Class(model_vqvae, test_loader, device, args.save_path)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ckpt_vqvae", type=str, default="//home/abghamtm/work/mnist_vqvae/checkpoint/vqvae/flat_vqvae_099.pt")
    parser.add_argument("--save_path", type=str, default="/home/abghamtm/work/mnist_vqvae/reconstructed_images/test")
    args = parser.parse_args()

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

