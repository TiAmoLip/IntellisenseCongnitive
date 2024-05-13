from sympy import arg
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from model import VAECnn, VAEMlp
import argparse
from methods import run_training_session
# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

parser = argparse.ArgumentParser()
# parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--type', type=str, default="mlp", choices=['cnn', 'mlp'])
parser.add_argument('--latent_size', type=int, default=2)

parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--base_channels', type=int, default=8)
parser.add_argument('--hiddens', type=int, default=100)
parser.add_argument('--step', type=float, default=0.1)

parser.add_argument('--lambda_kl', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--p', type=float, default=1)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument('--run_name', type=str, default=None)

args = parser.parse_args()
config = {

    'latent_size': args.latent_size,
    'num_layers': args.num_layers,
    'base_channels': args.base_channels,
    'lambda_kl': args.lambda_kl,
    'p': args.p,
    'lr': args.lr,
    "hiddens":args.hiddens,
    "run_name":args.run_name
}
if args.wandb:
    assert args.run_name is not None
    import wandb
    wandb.login(key="433d80a0f2ec170d67780fc27cd9d54a5039a57b")
    wandb.init(
        project="VAE",
        config=config,
        name=args.run_name
    )



# MNist Data Loader
batch_size=50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.type == "cnn":
    model = VAECnn(args).to(device)
else:
    model = VAEMlp(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print(config)


run_training_session(args = args,model=model,train_loader=train_loader,
                     test_loader=test_loader,config=config,optimizer=optimizer,
                     continue_training_path = None,device=device)

if args.latent_size == 1:
    x = torch.arange(-10, 10, args.step)
    res = model.decoder.decode(x.reshape(-1,1).float().to(device))
    save_image(res.view(res.shape[0], 1, 28, 28), f'./images/{args.run_name}/output' + '.png', nrow=int(x.shape[0]**0.5+0.001))

elif args.latent_size == 2:
    x = torch.arange(-5, 5, args.step)
    y = torch.arange(-5, 5, args.step)
    samples = []
    for i in x:
        for j in y:
            z = torch.tensor([i, j]).float().to(device)
            samples.append(z)

    samples = torch.stack(samples)
    res = model.decoder.decode(samples.to(device))

    save_image(res.view(res.shape[0], 1, 28, 28), f'./images/{args.run_name}/output' + '.png', nrow=x.shape[0])

if args.wandb:
    wandb.finish()