from model import VAEMlp,VAECnn
import torch
from torch import nn
import argparse
from torchvision.utils import save_image
parser = argparse.ArgumentParser(description='VAE generate')

# python generate.py --model mlp --model_path checkpoints/l2_n4_h200_kl1-3_p1.pth --latent_size 2 --hiddens 200 --lambda_kl 0.001 --p 1 --num_layers 4

parser.add_argument('--model', type=str, default='mlp', help='mlp or cnn')
parser.add_argument('--model_path', type=str, default='model/mlp.pth', help='model path')

parser.add_argument('--step', type=float, default=1)

parser.add_argument('--output', type=str, default='output', help='output path')
parser.add_argument('--latent_size', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--base_channels', type=int, default=8)
parser.add_argument('--hiddens', type=int, default=100)
parser.add_argument('--lambda_kl', type=float, default=1e-3)
parser.add_argument('--p', type=float, default=0.9)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.model == 'mlp':
    model = VAEMlp(args).to(device)
    
else:
    model = VAECnn(args).to(device)
model.load_state_dict(torch.load(args.model_path))
import os
# os.makedirs(args.output, exist_ok=True)
outDir = args.output if "png" in args.output else args.output + ".png"

if args.latent_size == 2:
    x = torch.arange(-5, 5, args.step)
    y = torch.arange(-5, 5, args.step)

    samples = []
    for i in x:
        for j in y:
            z = torch.tensor([i, j]).float().to(device)
            samples.append(z)

    samples = torch.stack(samples)

elif args.latent_size == 1:
    x1 = torch.arange(-17, -2, 5)
    x2 = torch.arange(-2, 2, args.step)
    x3 = torch.arange(2, 17, 5)
    x = torch.cat((x1, x2, x3))
    # x = torch.arange(100,150,1)
    # x = torch.cat((x, torch.arange(-150,-100,1)))
    samples = x.reshape(-1,1).float().to(device)
    

res = model.decoder.decode(samples.to(device))
save_image(res.view(res.shape[0], 1, 28, 28), outDir, nrow=x.shape[0] if args.latent_size==2 else int((x.shape[0]+0.001)**0.5))
# print(res.shape)
