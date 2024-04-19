mode=$1

# with rope and adam
if [ $mode == "rope_adam" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --wandb --run_name L_rope_adam --lr 0.003
fi

if [ $mode == "adam" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --wandb --run_name L_adam --lr 0.003
fi

if [ $mode == "sgd" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --wandb --run_name L_sgd --lr 0.003
fi

if [ $mode == "rope_sgd" ]; then
    python main.py --data gigaspeech --cuda --epochs 40 --model L --rope --wandb --run_name L_rope_sgd --lr 0.003
fi

