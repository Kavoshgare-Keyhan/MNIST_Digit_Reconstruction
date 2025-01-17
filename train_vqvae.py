import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils 
from torchvision.datasets import ImageNet

from tqdm import tqdm

from vqvae import FlatVQVAE
# from scheduler import CycleScheduler
from torch.optim.lr_scheduler import CyclicLR
import distributed as dist

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import neptune.new as neptune

os. nice (19)

# run = neptune.init_run(
#     project="",
#     api_token="==",
#     capture_stdout=False,
#     capture_stderr=False,
# )

def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.35
    diversity_loss_weight = 0.0001
    sample_size = 5

    mse_sum = 0
    mse_n = 0

    total_loss = 0
    total_recon_loss = 0
    total_quantization_loss = 0
    num_samples = 0
    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)
        out, latent_loss, diversity_loss, codebook_usage = model(img)
        recon_loss = criterion(out, img)

        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss

        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            # run["train/mse"].log(recon_loss.item())
            # run["train/latent"].log(latent_loss.item())
            # run["train/epoch"].log(epoch + 1)
            # run["train/num_used_codebooks"].log(codebook_usage)
            

            if i % 100 == 0:
                model.eval()
                sample = img[:sample_size]
                with torch.no_grad():
                    out, _ ,_,_ = model(sample)
                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"/home/abghamtm/work/mnist_vqvae/reconstructed_images/train/flat_vqvae_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                model.train()
    
        total_loss += loss.item() * img.size(0)
        total_recon_loss += recon_loss.item() * img.size(0)
        total_quantization_loss += latent_loss.item() * img.size(0)
        num_samples += img.size(0) 

    avg_loss = total_loss/num_samples
    avg_recons_loss = total_recon_loss/num_samples
    avg_quantization_loss = total_quantization_loss/num_samples
    print(
            f"Step {i}, Epoch {epoch + 1}: "
            f"Avg Loss = {avg_loss:.5f}, "
            f"Avg Reconstruction Loss = {avg_recons_loss:.5f}, "
            f"Avg Quantization Loss = {avg_quantization_loss:.5f}"
        )


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    torch.cuda.set_device(1)  # Use GPU 1 (if desired)

    # 2. Clear any cached memory on the selected GPU
    torch.cuda.empty_cache()

    # 3. Ensure device variable is correctly set
    device = "cuda" if torch.cuda.is_available() else "cpu"


    args.distributed = dist.get_world_size() > 1

    # Transform to normalize data and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor (0-1 range)
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)

    model = FlatVQVAE().to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # run["train/lr"].log(args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CyclicLR(
        optimizer, 
        base_lr=args.lr * 0.1, 
        max_lr=args.lr, 
        step_size_up=len(train_loader) * args.epoch * 0.05, 
        mode="triangular",
        cycle_momentum=False)
    x=0
    for i in range(args.epoch):
        train(i, train_loader, model, optimizer, scheduler, device)
        x=i
        if dist.is_primary():
            torch.save(model.state_dict(), f"/home/abghamtm/work/mnist_vqvae/checkpoint/vqvae/flat_vqvae_{str(i + 0).zfill(3)}.pt")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
