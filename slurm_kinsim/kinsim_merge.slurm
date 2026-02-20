#!/bin/bash
#SBATCH --job-name=KinSim_Train
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pshort_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_merge_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

kinsim dictionary merge /data/users/ndutilleux/KinSim/dictionary_test/shards/ /data/users/ndutilleux/KinSim/dictionary_test/master_dict.plk
