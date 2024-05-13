#python main.py --latent_size 2 --num_layers 2 --base_channels 8 --lambda_kl 0.001 --p 0.9 --wandb True --run_name l2_n2_c8_kl-3_p9
#python main.py --latent_size 2 --num_layers 2 --base_channels 8 --lambda_kl 0.0001 --p 0.9 --wandb True --run_name l2_n2_c8_kl-4_p9
#python main.py --latent_size 2 --num_layers 2 --base_channels 4 --lambda_kl 0.001 --p 0.9 --wandb True --run_name l2_n2_c4_kl-3_p9


# python main.py --latent_size 1 --num_layers 2 --base_channels 16 --lambda_kl 0.01 --p 0.9 --wandb True --run_name l1_n2_c16_kl-2_p9
# python main.py --latent_size 1 --num_layers 2 --base_channels 16 --lambda_kl 0.001 --p 0.9 --wandb True --run_name l1_n2_c16_kl-3_p9
# python main.py --latent_size 1 --num_layers 2 --base_channels 32 --lambda_kl 0.01 --p 0.9 --wandb True --run_name l1_n2_c32_kl-2_p9








# python main.py --latent_size 1 --num_layers 2 --base_channels 16 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l1_n2_b16_kl1-3_p1
# python main.py --latent_size 1 --num_layers 4 --base_channels 16 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l1_n4_b16_kl1-3_p1
# python main.py --latent_size 1 --num_layers 3 --base_channels 16 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l1_n3_b16_kl1-3_p1
# python main.py --latent_size 1 --num_layers 2 --base_channels 16 --lambda_kl 0.01 --p 1 --type cnn --wandb True --run_name l1_n2_b16_kl1-2_p1
# python main.py --latent_size 1 --num_layers 2 --base_channels 16 --lambda_kl 0.005 --p 1 --type cnn --wandb True --run_name l1_n2_b16_kl5-3_p1


# python main.py --latent_size 2 --num_layers 2 --base_channels 16 --lambda_kl 1 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl1_p1
# python main.py --latent_size 2 --num_layers 2 --base_channels 16 --lambda_kl 2 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl2_p1
# python main.py --latent_size 2 --num_layers 2 --base_channels 16 --lambda_kl 3 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl3_p1


# python main.py --latent_size 1 --num_layers 3 --hiddens 300 --lambda_kl 0.005 --p 1 --type mlp --wandb True --run_name l1_n3_h300_kl5-3_p1
# python main.py --latent_size 1 --num_layers 3 --hiddens 300 --lambda_kl 0.001 --p 1 --type mlp --wandb True --run_name l1_n3_h300_kl1-3_p1

# python main.py --latent_size 1 --num_layers 3 --base_channels 8 --lambda_kl 0.001 --p 1 --type mlp --wandb True --run_name l1_n3_b8_kl1-3_p1
# python main.py --latent_size 1 --num_layers 3 --base_channels 32 --lambda_kl 0.001 --p 1 --type mlp --wandb True --run_name l1_n3_b32_kl1-3_p1
# python main.py --latent_size 1 --num_layers 3 --base_channels 8 --lambda_kl 0.005 --p 1 --type mlp --wandb True --run_name l1_n3_b8_kl5-3_p1
# python main.py --latent_size 1 --num_layers 3 --base_channels 32 --lambda_kl 0.005 --p 1 --type mlp --wandb True --run_name l1_n3_b32_kl5-3_p1

python main.py --latent_size 2 --num_layers 2 --base_channels 8 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n2_b8_kl1-3_p1

python main.py --latent_size 2 --num_layers 2 --base_channels 16 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl1-3_p1

python main.py --latent_size 2 --num_layers 2 --base_channels 32 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n2_b32_kl1-3_p1

python main.py --latent_size 2 --num_layers 3 --base_channels 8 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n3_b8_kl1-3_p1

python main.py --latent_size 2 --num_layers 3 --base_channels 16 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n3_b16_kl1-3_p1

python main.py --latent_size 2 --num_layers 4 --base_channels 8 --lambda_kl 0.001 --p 1 --type cnn --wandb True --run_name l2_n4_b8_kl1-3_p1 

# python main.py --latent_size 2 --num_layers 2 --hiddens 200 --lambda_kl 2 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl2_p1
# python main.py --latent_size 2 --num_layers 2 --hiddens 200 --lambda_kl 3 --p 1 --type cnn --wandb True --run_name l2_n2_b16_kl3_p1
