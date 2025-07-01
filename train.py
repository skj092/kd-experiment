import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import random
import wandb
from dotenv import load_dotenv

load_dotenv()

# -------------------
# Set reproducibility seed
# -------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------
# Configuration
# -------------------
batch_size = 100
initial_lr = 10.0
decay_factor = 0.998
pi = 0.5
pf = 0.99
T = 500
epochs = 3000
l_max = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

architectures = {
    "A": [784, 800, 800, 10],
    "B": [784, 1200, 1200, 10],
    "C": [784, 2000, 2000, 10],
    "D": [784, 1200, 1200, 1200, 10],
}

arch_name = "A"
arch = architectures[arch_name]

# -------------------
# Initialize WandB
# -------------------
wandb.init(project="mnist_dropout", name=f"dropout_arch_{arch_name}", config={
    "architecture": arch,
    "batch_size": batch_size,
    "initial_lr": initial_lr,
    "decay_factor": decay_factor,
    "pi": pi,
    "pf": pf,
    "T": T,
    "epochs": epochs,
    "l_max": l_max,
    "seed": seed
})

# -------------------
# Dataset
# -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------
# Model definition
# -------------------
class DropoutNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.out = nn.Linear(layers[-2], layers[-1])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.dropout(x, p=0.2, training=self.training)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.out(x)
        return x

model = DropoutNet(arch).to(device)

criterion = nn.CrossEntropyLoss()

# -------------------
# Manual momentum & update buffer
# -------------------
prev_updates = {}
for name, param in model.named_parameters():
    prev_updates[name] = torch.zeros_like(param.data)

lr = initial_lr
momentum = pi

# -------------------
# Training loop
# -------------------
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for data, target in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)

        optimizer = None  # avoid using optimizer.step()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Weight constraint
            for layer in model.layers:
                w = layer.weight
                norm_sq = torch.sum(w.data ** 2, dim=1, keepdim=True)
                exceed = norm_sq > l_max
                scaling = torch.sqrt(l_max / (norm_sq + 1e-8))
                scaling = torch.where(exceed, scaling, torch.ones_like(scaling))
                w.data *= scaling

            # Manual momentum update
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                d_p = param.grad.data
                update = momentum * prev_updates[name] - (1 - momentum) * lr * d_p
                prev_updates[name] = update
                param.data += update

    # Decay learning rate
    lr *= decay_factor

    # Update momentum linearly
    if epoch < T:
        momentum = pi + (pf - pi) * (epoch / T)
    else:
        momentum = pf

    # Evaluate on test
    if epoch % 100 == 0 or epoch == 1:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader.dataset)

        print(f"Epoch {epoch}, Test Acc: {acc:.2f}%, Avg Train Loss: {avg_train_loss:.4f}, LR: {lr:.5f}, Momentum: {momentum:.3f}")

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "test_accuracy": acc,
            "avg_train_loss": avg_train_loss,
            "learning_rate": lr,
            "momentum": momentum
        })

print("Training completed.")
wandb.finish()
