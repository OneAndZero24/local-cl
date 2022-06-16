#!/bin/bash

#SBATCH --job-name=interval_cl
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task 10

source .env

trap "kill 0" INT
seed=2001
lr=0.1

# SGD
# for seed in 2001 2002 2003; do
#   for offline in False; do
#     for scenario in INC_TASK; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
#         cfg.learning_rate=$lr tags=["20220531_cifar20x5_alexnet_sgd_${scenario}_3seeds"]
#     done
#   done
# done


# Adam
# for lr in 1e-3 5e-4 1e-4 5e-5; do
#   for offline in False; do
#     for scenario in INC_CLASS INC_TASK INC_DOMAIN; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100_20tasks cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
#         cfg.learning_rate=$lr cfg.optimizer=ADAM tags=["20220410_cifar100_alexnet_adam_${scenario}"]
#     done
#   done
# done

## ADAM
# for offline in True; do
#   for seed in 2001 2002 2003 2004 2005; do
#     for scenario in INC_TASK INC_DOMAIN INC_CLASS; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar100 cfg.optimizer=ADAM cfg.learning_rate=0.01 \
#         cfg.seed=${seed} cfg.scenario=${scenario} cfg.offline=${offline} \
#         tags=["20220302_rerun_drop_last_adam_${scenario}"]
#     done
#   done
# done

# EWC
# for seed in 2001 2002 2003; do
#   for lambda in 5e-4; do
#     #    for optimizer in SGD ADAM; do
#     for optimizer in SGD; do
#       for scenario in INC_TASK; do
#         singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=EWC \
#           cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
#           cfg.scenario=${scenario} \
#           tags=["20220531_cifar20x5_alexnet_ewc_${scenario}_3seeds"]
#       done
#     done
#   done
# done
# 
# # EWCOnline
# for seed in 2001 2002 2003; do
#   for lambda in 5e-2; do
#     #    for optimizer in SGD ADAM; do
#     for optimizer in SGD; do
#       for scenario in INC_TASK; do
#         singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=EWC \
#           cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} cfg.ewc_mode=online \
#           cfg.scenario=${scenario} \
#           tags=["20220531_cifar20x5_alexnet_ewconline_${scenario}_3seeds"]
#       done
#     done
#   done
# done
# 
# # L2
# for seed in 2001 2002 2003; do
#   for lambda in 5e-3; do
#     #    for optimizer in SGD ADAM; do
#     for optimizer in SGD; do
#       for scenario in INC_TASK; do
#         singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=L2 \
#           cfg.seed=${seed} cfg.optimizer=${optimizer} cfg.ewc_lambda=${lambda} \
#           cfg.scenario=${scenario} \
#           tags=["20220531_cifar20x5_alexnet_l2_${scenario}_3seeds"]
#       done
#     done
#   done
# done
# 
# # 
# LWF
for seed in 2001 2002 2003; do
  for temperature in 1; do
    for alpha in 0.5; do
      for scenario in INC_TASK; do
        singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=LWF cfg.seed=${seed} cfg.lwf_alpha=${alpha} \
          cfg.lwf_temperature=${temperature} cfg.scenario=${scenario} \
          tags=["20220531_cifar20x5_alexnet_lwf_${scenario}_3seeds"]
      done
    done
  done
done
# 
# 
# # 
# # 
# # ## MAS
# for seed in 2001 2002 2003; do
# #  for lambda in 0.001; do
#    for lambda in 5e-4; do
#     for scenario in INC_TASK; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=MAS cfg.seed=${seed} cfg.ewc_lambda=${lambda} \
#         cfg.scenario=${scenario} \
#            tags=["20220531_cifar20x5_alexnet_mas_${scenario}_3seeds"]
#     done
#   done
# done
# # 
# # 
# # ## Synaptic Intelligence
# for seed in 2001 2002 2003; do
#   for lambda in 0.1; do
#      for scenario in INC_TASK; do
#       singularity exec --nv $SIF_PATH python3.9 train.py cfg=default_cifar20x5 cfg.strategy=SI \
#         cfg.seed=${seed} cfg.si_lambda=${lambda} cfg.scenario=${scenario} \
#            tags=["20220531_cifar20x5_alexnet_si_${scenario}_3seeds"]
#     done
#   done
# done
