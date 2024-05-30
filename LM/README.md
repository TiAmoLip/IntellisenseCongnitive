# Word-level Language Modeling using RNN and Transformer

Sota LSTM(58.49M parameters, 105 test ppl):
```bash
python main.py --data gigaspeech --cuda --epochs 100 --model LSTM --lr 20 --bptt 250 --log-interval 100 --hiddens 1120 --embed_size 384 --nlayers 2 --optimizer 1 --batch_size 30
```

Sota Transformer(60M parameters):
```bash
python main.py --data gigaspeech --cuda --epochs 70 --model Transformer --lr 0.5 --bptt 150 --log-interval 100 --hiddens 2048 --embed_size 468 --optimizer 1 --nlayers 12 --batch_size 30
```