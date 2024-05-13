To produce the required images when latent size is 1 or 2, use command:

```bash
# latent size 2
python generate.py --model mlp --model_path checkpoints/l2_n3_h400_kl1-3_p1.pth --latent_size 2 --hiddens 400 --lambda_kl 0.001 --p 1 --num_layers 3 --step 0.5
```


```bash
 python generate.py --model mlp --model_path checkpoints/l1_n3_h300_kl1-3_p1.pth --latent_size 1 --hiddens 300 --lambda_kl 0.005 --p 1 --num_layers 3 --step 0.025
```


The `output1.png` and `output2.png` are the output files of the above commands.