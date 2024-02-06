#!/bin/bash


#SBATCH --job-name TFG                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH -w dionisio

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

#SBATCH --mem=110G                         # RAM

module load rootless-docker # Obligatorio
start_rootless_docker.sh # Obligatorio


docker run --gpus all --rm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --env CUDA_VISIBLE_DEVICES=0 \
    --workdir /$USER -v /mnt/homeGPU/$USER/:/$USER \
    --shm-size=1G \
    -e HOME=/$USER xehartnort/pygpu:latest /bin/bash -c "bash /mariogmarq/flex-block/install_deps.sh && cd /mariogmarq/flex-block/flexBlock/experiments/baseline/celeba && python celeba.py"



stop_rootless_docker.sh