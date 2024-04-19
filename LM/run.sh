# with rope and adam
python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --wandb --run_name L_rope_adam

# without rope but adam
python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --wandb --run_name L_adam

# only sgd
python main.py --data gigaspeech --cuda --epochs 40 --model L --wandb --run_name L_sgd

# larger parameters:
python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --hidden_size 2048 --num_layers 16 --lr 0.01 --wandb --run_name L_rope_adam_2048hid_16layers

# larger parameters:
python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --hidden_size 2048 --num_layers 16 --lr 0.01 --nhead 16 --wandb --run_name L_rope_adam_2048hid_16layers_16nhead

# larger parameters:
python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --hidden_size 2048 --num_layers 12 --lr 0.01 --nhead 16 --embed_size 512 --wandb --run_name L_rope_adam_2048hid_12layers_16nhead_512embed