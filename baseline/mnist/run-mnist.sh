#!/bin/bash


#SBATCH --job-name TFG                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH -w hera

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

        

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/mariogmarq/flpy310


python mnist.py      


mail -s "Proceso finalizado" mariogarciamarq@gmail.com <<< "El proceso ha finalizado"