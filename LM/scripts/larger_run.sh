mode=$1
if [ $mode == "first" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --hiddens 2048 --nlayers 16 --lr 0.003 --wandb --run_name L_rope_adam_2048hid_16layers
fi

if [ $mode == "second" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --hiddens 2048 --nlayers 16 --lr 0.003 --nhead 16 --wandb --run_name L_adam_2048hid_16layers_16nhead
fi

if [ $mode == "third" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --hiddens 2048 --nlayers 12 --lr 0.003 --nhead 16 --embed_size 384 -bptt 64 --wandb --run_name L_rope_adam_2048hid_12layers_16nhead_512embed
fi

# larger parameters:

# larger parameters:
