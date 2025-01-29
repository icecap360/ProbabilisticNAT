#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py --config configs/AutoNAT_L_1.yaml --mode pretrain --beta_alpha_beta 12 3 --output_dir output/train1