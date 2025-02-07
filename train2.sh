#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
python -m torch.distributed.run --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_mrfsampler.py --config configs/AutoNAT_mrfsampler.yaml --mode pretrain --beta_alpha_beta 12 3 --output_dir output/train2
