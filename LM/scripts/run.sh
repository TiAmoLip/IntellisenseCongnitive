# mode=$1

# # with rope and adam
# if [ $mode == "rope_adam" ]; then
#     python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --rope --wandb --run_name L_rope_adam --lr 0.003
# fi

# if [ $mode == "adam" ]; then
#     python main.py --data gigaspeech --cuda --epochs 40 --model L --optimizer --wandb --run_name L_adam --lr 0.003
# fi

# if [ $mode == "sgd" ]; then
#     python main.py --data gigaspeech --cuda --epochs 40 --model L --wandb --run_name L_sgd --lr 0.003
# fi

# if [ $mode == "rope_sgd" ]; then
#     python main.py --data gigaspeech --cuda --epochs 40 --model L --rope --wandb --run_name L_rope_sgd --lr 0.003
# fi


# python main.py --data wikitext --cuda --epochs 10 --wandb --run_name LSTM_wiki --model LSTM --bptt 150 --hiddens 800 --nlayers 2 --embed_size 400 --lr 20 --batch_size 20 --log-interval 100
# python main.py --data wikitext --cuda --epochs 10 --model L --optimizer --rope --hiddens 1024 --nlayers 16 --lr 0.05 --nhead 8 --embed_size 256 --bptt 150 --wandb --run_name rope_l_wiki --log-interval 100 --batch_size 20
# with rope and adam
# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name embed_256 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 384 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name embed_384 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 384 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name embed_128 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 128 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name embed_64 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 64 --nlayers 2


# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name bptt_150 --lr 20 --bptt 150 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name bptt_100 --lr 20 --bptt 100 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name clip05 --clip 0.5 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model LSTM --wandb --run_name clip1 --clip 1 --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 100 --model GRU --wandb --run_name GRURawSetting --lr 20 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 2

# python main.py --data gigaspeech --cuda --epochs 25 --model L --wandb --run_name BigLRwithClip --lr 10 --bptt 50 --log-interval 100 --hiddens 1120 --embed_size 256 --nlayers 16
# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name TransformerRawSetting --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 24
# python main.py --data gigaspeech --cuda --epochs 100 --model Decoder --wandb --run_name DecoderRawSetting --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 16


# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name Thidden2048 --lr 0.5 --bptt 50 --log-interval 100 --hiddens 2048 --embed_size 256 --nlayers 16

# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name Thidden1024 --lr 0.5 --bptt 50 --log-interval 100 --hiddens 1024 --embed_size 256 --nlayers 16

# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name TransformerL20 --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 20
# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name TransformerL12 --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 12
# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name TransformerL24Moment --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 24 --optimizer

# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name Tbptt100 --lr 0.5 --bptt 100 --log-interval 100 --hiddens 3072 --embed_size 256 --nlayers 24 --optimizer --batch_size 30

# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name TEmbed128 --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 128 --nlayers 12 --optimizer --batch_size 30


# python main.py --data gigaspeech --cuda --epochs 100 --model Transformer --wandb --run_name embed512Momentbptt250 --lr 0.5 --bptt 250 --log-interval 100 --hiddens 2048 --embed_size 512 --nlayers 12 --optimizer --batch_size 20

# # step 0:
# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed512bp150Moment --lr 0.5 --bptt 150 --log-interval 100 --hiddens 2048 --embed_size 512 --optimizer 1 --nlayers 12 --batch_size 30


# # step 1
# python main.py --data gigaspeech --cuda --epochs 100 --model LSTM --wandb --run_name embed384Momentbp150 --lr 20 --bptt 150 --log-interval 100 --hiddens 1120 --embed_size 384 --nlayers 2 --optimizer 1 --batch_size 30
# python main.py --data gigaspeech --cuda --epochs 100 --model LSTM --wandb --run_name embed384Momentbp250 --lr 20 --bptt 250 --log-interval 100 --hiddens 1120 --embed_size 384 --nlayers 2 --optimizer 1 --batch_size 30


# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed512bp150AdamW --lr 0.5 --bptt 150 --log-interval 100 --hiddens 2048 --embed_size 512 --optimizer 3 --nlayers 12 --batch_size 30


# # step 2
# python main.py --data wikitext --cuda --epochs 50 --model Transformer --wandb --run_name TWiki --lr 0.5 --bptt 50 --log-interval 100 --hiddens 3072 --embed_size 384 --optimizer 1 --nlayers 12 --batch_size 30
# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name BIGMODEL --lr 0.5 --bptt 150 --log-interval 100 --hiddens 3072 --embed_size 512 --optimizer 1 --nlayers 16 --batch_size 20

# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed768bp250Moment --lr 0.5 --bptt 250 --log-interval 100 --hiddens 2048 --embed_size 512 --optimizer 1 --nlayers 12 --batch_size 20

# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed384bp150 --lr 0.5 --bptt 150 --log-interval 100 --hiddens 3072 --embed_size 384 --nlayers 12 --batch_size 30
# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed384bp250 --lr 0.5 --bptt 250 --log-interval 100 --hiddens 3072 --embed_size 384 --nlayers 12 --batch_size 30


# # step 3
# python main.py --data gigaspeech --cuda --epochs 100 --model L --wandb --run_name Lembed768bp150 --lr 0.01 --bptt 150 --log-interval 100 --hiddens 2048 --embed_size 768 --nlayers 10 --batch_size 30
# python main.py --data gigaspeech --cuda --epochs 100 --model L --wandb --run_name Lembed768bp250 --lr 0.01 --bptt 250 --log-interval 100 --hiddens 2048 --embed_size 768 --nlayers 10 --batch_size 30

# # step 4
# python main.py --data gigaspeech --cuda --epochs 50 --model Transformer --wandb --run_name Tembed384bp150Adam --optimzer 2 --lr 0.5 --bptt 150 --log-interval 100 --hiddens 3072 --embed_size 384 --nlayers 12 --batch_size 30
