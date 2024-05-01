from sympy import arg
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from model import VAE
import argparse
from methods import run_training_session
# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

parser = argparse.ArgumentParser()
# parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--latent_size', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--base_channels', type=int, default=8)
parser.add_argument('--lambda_kl', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--p', type=float, default=0.9)
args = parser.parse_args()
# MNist Data Loader
batch_size=50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {

    'latent_size': args.latent_size,
    'num_layers': args.num_layers,
    'base_channels': args.base_channels,
    'lambda_kl': args.lambda_kl,
    'p': args.p,
    'lr': args.lr,
}
model = VAE(**config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




run_training_session(model=model,train_loader=train_loader,
                     test_loader=test_loader,config=config,optimizer=optimizer,
                     continue_training_path = None,device=device)
