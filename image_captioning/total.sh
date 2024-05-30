#!/bin/bash

#SBATCH --job-name=train
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# __conda_setup="$('/lustre/home/acct-stu/stu282/Tools/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/lustre/home/acct-stu/stu282/Tools/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/lustre/home/acct-stu/stu282/Tools/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/lustre/home/acct-stu/stu282/Tools/miniconda3/bin:$PATH"
#     fi
# fi

# export PATH="/lustre/home/acct-stu/stu282/Tools/miniconda3/envs/py3.10.11:$PATH"
# conda config --set auto_activate_base false
# conda init
# conda activate py3.10.11

module load miniconda3
source activate py3.10

python download_punkt.py
cd /lustre/home/acct-stu/stu343/IntellisenseCongnitive/image_captioning


# 注意，由于我懒得改每个config的名字，这里的512实际上是256，但最后的输出路径会改成256
python main.py train_evaluate --config_file configs/c3_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/c5_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/c10_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/c15_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/c20_g_e256_a256_d512_adam.yaml

python main.py train_evaluate --config_file configs/l10_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/l20_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/l40_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/l60_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/l80_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/l100_g_e256_a256_d512_adam.yaml

python main.py train_evaluate --config_file configs/s5_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/s10_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/s15_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/s20_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/s30_g_e256_a256_d512_adam.yaml
python main.py train_evaluate --config_file configs/s40_g_e256_a256_d512_adam.yaml





# python main.py evaluate --config_file configs/c10_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/c15_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/c20_g_e256_a256_d512_adam.yaml

# python main.py evaluate --config_file configs/l10_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/l20_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/l40_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/l60_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/l80_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/l100_g_e256_a256_d512_adam.yaml


# python main.py evaluate --config_file configs/s15_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/s20_g_e256_a256_d512_adam.yaml
# python main.py evaluate --config_file configs/s40_g_e256_a256_d512_adam.yaml
