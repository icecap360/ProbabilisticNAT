#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.run --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py --config configs/AutoNAT_L.yaml --mode pretrain --beta_alpha_beta 12 3 --output_dir output/train0