#!/bin/bash

#SBATCH -J SimonLentz
#SBATCH -p gpu
#SBATCH --account=uo1075
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=mg206
#SBATCH --time=12:00:00
#SBATCH --mem=64G

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity exec --bind /work/uo1075/u301617/ --nv /work/uo1075/u301617/Master-Arbeit/pytorch_gpu_new.sif \
<<<<<<< HEAD
 python train.py
 --save_part 'part_2' --mask_year '2020' --image_year 'r8_12'

=======
 python train.py \
 --mask_year '2004_2020' --im_year 'r8_9' --save_part 'part_2'
>>>>>>> 40b72786acf56c8e6678ced7bdfa56c064883fe3
