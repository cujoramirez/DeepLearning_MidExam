"""
COMP6826001 Deep Learning - Problem 2: Pine Tree GAN (DCGAN)
Deep Convolutional GAN with BCE loss
Run: python case2_2.py
"""

import os
import gc
import copy
import json
import hashlib
import random
from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

STUDENT_ID = "2702268725"
STUDENT_NAME = "Gading Aditya Perdana"

LATENT_DIM = 100
IMG_CHANNELS = 3
IMG_SIZE = 32
GEN_FEATURES = 64
DISC_FEATURES = 64
G_LR = 0.0002
D_LR = 0.0002
BATCH_SIZE = 225
REAL_LABEL = 1.0
FAKE_LABEL = 0.0
BETA1 = 0.5
BETA2 = 0.999
GAN_EPOCHS = 3000
SAVE_INTERVAL = 10

CLS_EPOCHS = 25
CLS_LR = 2e-4
NUM_GENERATED = 500
VAL_SPLIT = 0.1
COMPARISON_COUNT = 10

GAN_MEAN = (0.5, 0.5, 0.5)
GAN_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR100_CLASSES = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm_tree", 57: "pear", 58: "pickup_truck", 59: "pine_tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
    90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
    95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm",
}


def setup_reproducibility(student_id: str):
    hash_value = int(hashlib.md5(student_id.encode("utf-8")).hexdigest(), 16)
    class_id = hash_value % 100
    seed_value = hash_value % (2**31)
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return class_id, seed_value


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(GAN_MEAN).view(-1, 1, 1)
    std = torch.tensor(GAN_STD).view(-1, 1, 1)
    return x * std + mean


def prepare_data(class_id: int, data_root: Path, seed: int, val_split: float = VAL_SPLIT):
    print(f"\n{'='*60}")
    print("DATA PREPARATION")
    print(f"{'='*60}")

    aug_transforms = [transforms.RandomHorizontalFlip()]
    if IMG_SIZE == 32:
        aug_transforms.append(transforms.RandomCrop(IMG_SIZE, padding=4, padding_mode='reflect'))
    else:
        aug_transforms.append(transforms.Resize(IMG_SIZE))
    aug_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(GAN_MEAN, GAN_STD),
    ])

    train_transform = transforms.Compose(aug_transforms)
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(GAN_MEAN, GAN_STD),
    ])

    data_root.mkdir(exist_ok=True)
    train_full = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    train_eval = datasets.CIFAR100(root=data_root, train=True, download=False, transform=eval_transform)
    test_full = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_transform)

    print(f"Filtering class: {CIFAR100_CLASSES[class_id]} (ID: {class_id})")

    train_indices = [idx for idx, label in enumerate(train_full.targets) if label == class_id]
    test_indices = [idx for idx, label in enumerate(test_full.targets) if label == class_id]

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_indices)
    val_count = max(1, int(len(shuffled) * val_split))
    val_indices = shuffled[:val_count].tolist()
    train_split_indices = shuffled[val_count:].tolist()

    pine_train = Subset(train_full, train_split_indices)
    pine_val = Subset(train_eval, val_indices)
    pine_eval = Subset(train_eval, train_indices)
    pine_test = Subset(test_full, test_indices)

    print(f"Pine-tree train samples: {len(train_split_indices)}")
    print(f"Pine-tree val samples: {len(val_indices)}")
    print(f"Pine-tree test samples: {len(test_indices)}")

    clear_memory()

    return pine_train, pine_val, pine_test, pine_eval


class Generator(nn.Module):
    """DCGAN Generator: 32x32 RGB images"""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, GEN_FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 8, GEN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 4, GEN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GEN_FEATURES * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(GEN_FEATURES * 2, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator: No Sigmoid for AMP, use BCEWithLogitsLoss"""
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, DISC_FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISC_FEATURES, DISC_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISC_FEATURES * 2, DISC_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISC_FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DISC_FEATURES * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img):
        return self.main(img).view(-1)


def evaluate_gan_losses(generator, discriminator, loader, device, criterion):
    generator.eval()
    discriminator.eval()
    g_losses, d_losses = [], []
    d_x_scores, d_g_z_scores = [], []
    with torch.no_grad():
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            label_real = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
            real_output = discriminator(real_imgs).view(-1)
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = generator(noise)
            fake_output = discriminator(fake).view(-1)
            d_loss_real = criterion(real_output, label_real)
            d_loss_fake = criterion(fake_output, label_fake)
            d_loss = d_loss_real + d_loss_fake
            g_loss = criterion(fake_output, label_real)
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_x_scores.append(torch.sigmoid(real_output).mean().item())
            d_g_z_scores.append(torch.sigmoid(fake_output).mean().item())
    generator.train()
    discriminator.train()
    return {
        'g_loss': float(np.mean(g_losses)) if g_losses else 0.0,
        'd_loss': float(np.mean(d_losses)) if d_losses else 0.0,
        'd_x': float(np.mean(d_x_scores)) if d_x_scores else 0.0,
        'd_g_z': float(np.mean(d_g_z_scores)) if d_g_z_scores else 0.0,
    }


def summarize_trend(values):
    if len(values) < 2:
        return "insufficient data"
    delta = values[-1] - values[0]
    direction = "decreased" if delta < 0 else "increased"
    magnitude = abs(delta)
    return f"{direction} by {magnitude:.3f} from {values[0]:.3f} to {values[-1]:.3f}"


def save_gan_analysis(train_losses, val_losses, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    analysis = []
    if train_losses:
        analysis.append(f"Generator train loss {summarize_trend(train_losses)}")
    if val_losses:
        analysis.append(f"Generator val loss {summarize_trend(val_losses)}")
    path.write_text("\n".join(analysis))


def save_comparison_grid(real_dataset, fake_images, outputs_dir: Path, count: int = COMPARISON_COUNT):
    figure_path = outputs_dir / "figures" / "real_vs_fake.png"
    (outputs_dir / "figures").mkdir(exist_ok=True, parents=True)
    real_tensors = []
    for i in range(min(count, len(real_dataset))):
        tensor = real_dataset[i][0]
        real_tensors.append(denormalize(tensor).clamp(0, 1))
    fake_tensors = [denormalize(img).clamp(0, 1) for img in fake_images[:count]]
    cols = min(count, len(real_tensors), len(fake_tensors))
    if cols == 0:
        return figure_path
    plt.figure(figsize=(2 * cols, 4))
    for idx in range(cols):
        plt.subplot(2, cols, idx + 1)
        plt.imshow(real_tensors[idx].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Real {idx+1}")
        plt.subplot(2, cols, cols + idx + 1)
        plt.imshow(fake_tensors[idx].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Fake {idx+1}")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return figure_path


def save_classifier_analysis(train_losses, val_losses, outputs_dir: Path):
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    analysis_path = figures_dir / "classifier_training_analysis.txt"
    text = [
        f"Classifier train loss {summarize_trend(train_losses)}" if train_losses else "",
        f"Classifier val loss {summarize_trend(val_losses)}" if val_losses else "",
    ]
    analysis_path.write_text("\n".join([line for line in text if line]))
    return analysis_path


def gan_hparam_search(pine_train, pine_val, device, outputs_dir: Path, base_epochs: int):
    trial_epochs = max(5, base_epochs // 10)
    search_space = [
        {"name": "baseline", "g_lr": G_LR, "d_lr": D_LR, "beta1": BETA1, "beta2": BETA2},
        {"name": "slower", "g_lr": G_LR * 0.5, "d_lr": D_LR * 0.5, "beta1": BETA1, "beta2": min(0.999, BETA2 + 0.05)},
        {"name": "fasterG", "g_lr": G_LR * 1.5, "d_lr": D_LR, "beta1": BETA1, "beta2": BETA2},
    ]
    val_loader = DataLoader(pine_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    results = []
    tuning_root = outputs_dir / "gan_tuning"
    tuning_root.mkdir(exist_ok=True, parents=True)
    for cfg in search_space:
        trial_dir = tuning_root / cfg['name']
        trial_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n[TUNING] Running GAN trial {cfg['name']} for {trial_epochs} epochs")
        gen, disc = train_gan(pine_train, pine_val, device, trial_dir, epochs=trial_epochs,
                              g_lr=cfg['g_lr'], d_lr=cfg['d_lr'], beta1=cfg['beta1'], beta2=cfg['beta2'],
                              log_prefix=f"[TUNE {cfg['name']}] ")
        criterion = nn.BCEWithLogitsLoss()
        metrics = evaluate_gan_losses(gen, disc, val_loader, device, criterion)
        result = {**cfg, **metrics}
        results.append(result)
    results_path = tuning_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda r: r['d_loss']) if results else search_space[0]
    print(f"[TUNING] Best GAN config: {best}")
    clear_memory()
    return best


def classifier_hparam_search(train_ds, val_ds, device, outputs_dir: Path, base_epochs: int):
    tuning_epochs = max(3, base_epochs // 5)
    search_space = [
        {"name": "baseline", "lr": CLS_LR, "weight_decay": 1e-4},
        {"name": "low_lr", "lr": CLS_LR * 0.5, "weight_decay": 1e-4},
        {"name": "high_wd", "lr": CLS_LR, "weight_decay": 5e-4},
    ]
    tuning_root = outputs_dir / "classifier_tuning"
    tuning_root.mkdir(exist_ok=True, parents=True)
    results = []
    for cfg in search_space:
        trial_dir = tuning_root / cfg['name']
        trial_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n[TUNING] Running classifier trial {cfg['name']} for {tuning_epochs} epochs")
        _, _, val_losses = train_classifier(train_ds, val_ds, device, trial_dir, epochs=tuning_epochs,
                                            lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                                            log_prefix=f"[TUNE {cfg['name']}] ", save_artifacts=False)
        final_val = val_losses[-1] if val_losses else float('inf')
        results.append({**cfg, "val_loss": final_val})
    results_path = tuning_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda r: r['val_loss']) if results else search_space[0]
    print(f"[TUNING] Best classifier config: {best}")
    clear_memory()
    return best


def train_gan(pine_train, pine_val, device, outputs_dir, resume_from=None, epochs: int = GAN_EPOCHS,
              g_lr: float = G_LR, d_lr: float = D_LR, beta1: float = BETA1, beta2: float = BETA2,
              log_prefix: str = ""):
    print(f"\n{'='*60}")
    print(f"{log_prefix}DCGAN TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Architecture: Deep Convolutional GAN")
    print(f"Loss: Binary Cross-Entropy")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"G_LR: {g_lr}, D_LR: {d_lr}")
    print(f"Beta1: {beta1}, Beta2: {beta2}")
    print(f"{'='*60}\n")
    
    figures_dir = outputs_dir / "figures"
    checkpoints_dir = outputs_dir / "checkpoints"
    figures_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    criterion = nn.BCEWithLogitsLoss()
    
    opt_G = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))
    opt_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    pine_loader = DataLoader(pine_train, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=torch.cuda.is_available())
    pine_val_loader = DataLoader(pine_val, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=torch.cuda.is_available())
    
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    G_losses, D_losses, D_x_scores, D_G_z_scores = [], [], [], []
    G_val_losses, D_val_losses = [], []
    start_epoch = 1
    
    if resume_from and resume_from.exists():
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        G_losses = checkpoint['g_losses']
        D_losses = checkpoint['d_losses']
        D_x_scores = checkpoint.get('d_x_scores', [])
        D_G_z_scores = checkpoint.get('d_g_z_scores', [])
        G_val_losses = checkpoint.get('g_val_losses', [])
        D_val_losses = checkpoint.get('d_val_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    print("Starting Training Loop...")
    for epoch in range(start_epoch, epochs + 1):
        epoch_g_losses, epoch_d_losses = [], []
        epoch_d_x, epoch_d_g_z = [], []
        
        pbar = tqdm(pine_loader, desc=f"Epoch {epoch}/{epochs}", ncols=120)
        
        for i, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            label_real = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
            
            # Train Discriminator
            opt_D.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    real_output = discriminator(real_imgs).view(-1)
                    errD_real = criterion(real_output, label_real)
                    noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                    fake = generator(noise)
                    fake_output = discriminator(fake.detach()).view(-1)
                    errD_fake = criterion(fake_output, label_fake)
                    errD = errD_real + errD_fake
                scaler.scale(errD).backward()
                scaler.step(opt_D)
                scaler.update()
            else:
                real_output = discriminator(real_imgs).view(-1)
                errD_real = criterion(real_output, label_real)
                noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                fake = generator(noise)
                fake_output = discriminator(fake.detach()).view(-1)
                errD_fake = criterion(fake_output, label_fake)
                errD = errD_real + errD_fake
                errD.backward()
                opt_D.step()
            
            D_x = torch.sigmoid(real_output).mean().item()
            D_G_z1 = torch.sigmoid(fake_output).mean().item()
            
            # Train Generator
            opt_G.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                    fake = generator(noise)
                    fake_output = discriminator(fake).view(-1)
                    errG = criterion(fake_output, label_real)
                scaler.scale(errG).backward()
                scaler.step(opt_G)
                scaler.update()
            else:
                noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
                fake = generator(noise)
                fake_output = discriminator(fake).view(-1)
                errG = criterion(fake_output, label_real)
                errG.backward()
                opt_G.step()
            
            D_G_z2 = torch.sigmoid(fake_output).mean().item()
            
            epoch_g_losses.append(errG.item())
            epoch_d_losses.append(errD.item())
            epoch_d_x.append(D_x)
            epoch_d_g_z.append(D_G_z2)
            
            if i % 50 == 0:
                pbar.set_postfix({
                    'D_loss': f'{errD.item():.3f}',
                    'G_loss': f'{errG.item():.3f}',
                    'D(x)': f'{D_x:.3f}',
                    'D(G(z))': f'{D_G_z2:.3f}'
                })
        
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        avg_d_x = np.mean(epoch_d_x)
        avg_d_g_z = np.mean(epoch_d_g_z)
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        D_x_scores.append(avg_d_x)
        D_G_z_scores.append(avg_d_g_z)
        val_metrics = evaluate_gan_losses(generator, discriminator, pine_val_loader, device, criterion)
        G_val_losses.append(val_metrics['g_loss'])
        D_val_losses.append(val_metrics['d_loss'])
        
        if epoch % SAVE_INTERVAL == 0 or epoch == 1 or epoch == epochs:
            print(f"\n{log_prefix}Epoch [{epoch}/{epochs}] D_loss: {avg_d_loss:.4f} (val {D_val_losses[-1]:.4f}) "
                  f"G_loss: {avg_g_loss:.4f} (val {G_val_losses[-1]:.4f}) "
                  f"D(x): {avg_d_x:.3f} D(G(z)): {avg_d_g_z:.3f}")
            
            with torch.no_grad():
                generator.eval()
                fake_samples = generator(fixed_noise).cpu()
                generator.train()
            
            fake_samples = (fake_samples + 1) / 2
            grid = vutils.make_grid(fake_samples, nrow=8, normalize=True)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
            plt.axis("off")
            plt.title(f"Epoch {epoch}")
            plt.savefig(figures_dir / f"samples_epoch_{epoch:04d}.png", dpi=150, bbox_inches="tight")
            plt.close()
            
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpoint_dict = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'g_losses': G_losses,
                'd_losses': D_losses,
                'd_x_scores': D_x_scores,
                'd_g_z_scores': D_G_z_scores,
                'g_val_losses': G_val_losses,
                'd_val_losses': D_val_losses,
            }
            if scaler:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)
            
            if epoch > SAVE_INTERVAL and epoch % 100 != 0:
                old_checkpoint = checkpoints_dir / f"checkpoint_epoch_{epoch-SAVE_INTERVAL:04d}.pt"
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            
            clear_memory()
    
    print("\nGenerating training plots...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(G_losses, label="G Train", alpha=0.8, linewidth=1.5)
    plt.plot(D_losses, label="D Train", alpha=0.8, linewidth=1.5)
    if G_val_losses:
        plt.plot(G_val_losses, label="G Val", linestyle='--')
    if D_val_losses:
        plt.plot(D_val_losses, label="D Val", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(D_x_scores, label="D(x) - Real", alpha=0.8, linewidth=1.5)
    plt.plot(D_G_z_scores, label="D(G(z)) - Fake", alpha=0.8, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Discriminator Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    window = 50
    if len(G_losses) >= window:
        g_ma = np.convolve(G_losses, np.ones(window)/window, mode='valid')
        d_ma = np.convolve(D_losses, np.ones(window)/window, mode='valid')
        plt.plot(g_ma, label="G Loss (MA)", alpha=0.8)
        plt.plot(d_ma, label="D Loss (MA)", alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Moving Average (window={window})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    diff_scores = [abs(x - y) for x, y in zip(D_x_scores, D_G_z_scores)]
    plt.plot(diff_scores, label="D(x) - D(G(z))", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Difference")
    plt.title("Score Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    final_dict = {
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_losses': G_losses,
        'd_losses': D_losses,
        'd_x_scores': D_x_scores,
        'd_g_z_scores': D_G_z_scores,
        'g_val_losses': G_val_losses,
        'd_val_losses': D_val_losses,
    }
    torch.save(final_dict, checkpoints_dir / "final_model.pt")
    save_gan_analysis(G_losses, G_val_losses, outputs_dir / "figures" / "gan_training_analysis.txt")
    
    print(f"\nTraining complete. Models saved to {checkpoints_dir}")
    return generator, discriminator


def generate_fake_images(generator, device, outputs_dir, count=512):
    print(f"\n{'='*60}")
    print(f"GENERATING {count} IMAGES")
    print(f"{'='*60}")
    
    generated_dir = outputs_dir / "generated"
    generated_dir.mkdir(exist_ok=True, parents=True)
    
    generator.eval()
    all_samples = []
    
    with torch.no_grad():
        for i in tqdm(range(0, count, BATCH_SIZE), desc="Generating"):
            batch_count = min(BATCH_SIZE, count - i)
            z = torch.randn(batch_count, LATENT_DIM, 1, 1, device=device)
            imgs = generator(z).cpu()
            all_samples.append(imgs)
    
    synth_images = torch.cat(all_samples, dim=0)[:count]
    
    for idx, tensor_img in enumerate(synth_images):
        np.save(generated_dir / f"pine_tree_fake_{idx:04d}.npy", tensor_img.numpy())
    
    print(f"Saved {len(synth_images)} images to {generated_dir}")
    clear_memory()
    
    return synth_images


class FakeOriginalClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        return self.backbone(x)


def prepare_classification_data(pine_dataset_eval, synth_images):
    print(f"\n{'='*60}")
    print("PREPARING CLASSIFICATION DATASET")
    print(f"{'='*60}")
    
    real_imgs = torch.stack([denormalize(pine_dataset_eval[i][0]) for i in range(len(pine_dataset_eval))])
    fake_imgs = torch.stack([denormalize(img) for img in synth_images])
    
    real_np = real_imgs.permute(0, 2, 3, 1).numpy()
    fake_np = fake_imgs.permute(0, 2, 3, 1).numpy()
    
    X = np.concatenate([real_np, fake_np], axis=0)
    y = np.concatenate([np.zeros(len(real_np)), np.ones(len(fake_np))], axis=0).astype(np.int64)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    imagenet_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def to_dataset(images, labels):
        tensors = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        for i in range(len(tensors)):
            tensors[i] = imagenet_norm(tensors[i])
        return TensorDataset(tensors, torch.from_numpy(labels))
    
    train_ds = to_dataset(X_train, y_train)
    val_ds = to_dataset(X_val, y_val)
    test_ds = to_dataset(X_test, y_test)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    del real_imgs, fake_imgs, real_np, fake_np, X, y, X_train, X_temp, X_val, X_test
    clear_memory()
    
    return train_ds, val_ds, test_ds


def train_classifier(train_ds, val_ds, device, outputs_dir, epochs: int = CLS_EPOCHS,
                     lr: float = CLS_LR, weight_decay: float = 1e-4,
                     log_prefix: str = "", save_artifacts: bool = True):
    print(f"\n{'='*60}")
    print("TRAINING CLASSIFIER")
    print(f"{'='*60}")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    
    model = FakeOriginalClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"{log_prefix}Epoch {epoch}/{epochs}", ncols=100):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"{log_prefix}Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        clear_memory()
    
    if save_artifacts:
        figures_dir = outputs_dir / "figures"
        figures_dir.mkdir(exist_ok=True, parents=True)
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Classifier Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / "classifier_losses.png", dpi=300, bbox_inches="tight")
        plt.close()
        torch.save(model.state_dict(), outputs_dir / "classifier_model.pt")
        print(f"Classifier saved")
    
    return model, train_losses, val_losses


def evaluate_classifier(model, test_ds, device, outputs_dir):
    print(f"\n{'='*60}")
    print("EVALUATING CLASSIFIER")
    print(f"{'='*60}")
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing", ncols=100):
            logits = model(X_batch.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outputs_dir / "figures" / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    with open(outputs_dir / "test_metrics.txt", "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"{cm}\n")


def main():
    parser = argparse.ArgumentParser(description="DCGAN Pine Tree Training")
    parser.add_argument("--skip-gan", action="store_true", help="Skip GAN training")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip classifier training")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=GAN_EPOCHS, help="Number of epochs")
    parser.add_argument("--class-id", type=int, help="Override CIFAR-100 class ID from get_class_to_work notebook")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Validation split fraction for GAN training")
    parser.add_argument("--comparison-count", type=int, default=COMPARISON_COUNT, help="Number of images per row when comparing real vs fake")
    parser.add_argument("--tune-classifier", action="store_true", help="Run quick hyperparameter tuning for the classifier before full training")
    parser.add_argument("--tune-gan", action="store_true", help="Run a lightweight GAN hyperparameter sweep using the validation split")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COMP6826001 - PROBLEM 2: PINE TREE DCGAN")
    print(f"{'='*60}")
    print(f"Student: {STUDENT_NAME}")
    print(f"ID: {STUDENT_ID}")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    class_id, seed = setup_reproducibility(STUDENT_ID)
    if args.class_id is not None:
        class_id = args.class_id % 100
        print(f"Overriding class ID via CLI: {class_id}")
    print(f"Class: {CIFAR100_CLASSES[class_id]} (ID: {class_id})")
    print(f"Seed: {seed}\n")
    
    outputs_dir = Path("problem2_outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    data_root = Path("cifar100-cache")
    pine_train, pine_val, pine_test, pine_eval = prepare_data(class_id, data_root, seed, val_split=args.val_split)
    
    if not args.skip_gan:
        resume_path = Path(args.resume) if args.resume else None
        gan_cfg = {"g_lr": G_LR, "d_lr": D_LR, "beta1": BETA1, "beta2": BETA2}
        if args.tune_gan:
            best_gan_cfg = gan_hparam_search(pine_train, pine_val, device, outputs_dir, args.epochs)
            for key in ('g_lr', 'd_lr', 'beta1', 'beta2'):
                if key in best_gan_cfg:
                    gan_cfg[key] = best_gan_cfg[key]
        generator, discriminator = train_gan(
            pine_train, pine_val, device, outputs_dir,
            resume_from=resume_path, epochs=args.epochs,
            g_lr=gan_cfg['g_lr'], d_lr=gan_cfg['d_lr'],
            beta1=gan_cfg['beta1'], beta2=gan_cfg['beta2'])
    else:
        print("Loading from checkpoint...")
        generator = Generator().to(device)
        checkpoint = torch.load(outputs_dir / "checkpoints" / "final_model.pt", map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator = None
    
    synth_images = generate_fake_images(generator, device, outputs_dir, count=NUM_GENERATED)
    comparison_path = save_comparison_grid(pine_eval, synth_images, outputs_dir, count=args.comparison_count)
    print(f"Saved real vs fake comparison grid to {comparison_path}")
    
    if not args.skip_classifier:
        train_ds, val_ds, test_ds = prepare_classification_data(pine_eval, synth_images)
        cls_cfg = {"lr": CLS_LR, "weight_decay": 1e-4}
        if args.tune_classifier:
            best_cfg = classifier_hparam_search(train_ds, val_ds, device, outputs_dir, CLS_EPOCHS)
            cls_cfg.update({k: best_cfg[k] for k in ('lr', 'weight_decay') if k in best_cfg})
        print(f"Classifier hyperparameters -> lr: {cls_cfg['lr']:.2e}, weight_decay: {cls_cfg['weight_decay']:.2e}")
        classifier, cls_train_losses, cls_val_losses = train_classifier(
            train_ds, val_ds, device, outputs_dir,
            epochs=CLS_EPOCHS, lr=cls_cfg['lr'], weight_decay=cls_cfg['weight_decay'])
        cls_analysis = save_classifier_analysis(cls_train_losses, cls_val_losses, outputs_dir)
        print(f"Saved classifier training analysis to {cls_analysis}")
        evaluate_classifier(classifier, test_ds, device, outputs_dir)
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"Outputs: {outputs_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()