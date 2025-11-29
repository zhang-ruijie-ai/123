#!/bin/bash
#SBATCH -J test # Slurm job name
#SBATCH --partition=speech
#SBATCH --output=InternVL2-8B—ablation.out
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=246000M

# --- 环境变量配置 ---
nvidia-smi
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_ENDPOINT=https://hf-mirror.com


python run_all_ablation.py \
    --model_path "/data/LLM_Model/InternVL2-8B" \
    --analysis_dir "/workspace/lizhou/LLMPenetration/multi-agent_hatespeech/multi-agent_enhance_Multi3Hate_new/analysis_files" \
    --dataset_path "/workspace/lizhou/LLMPenetration/multi-agent_hatespeech/multi-agent_enhance_Multi3Hate_new/train-00000-of-00001.parquet" \
    --output_dir "/workspace/lizhou/LLMPenetration/multi-agent_hatespeech/multi-agent_enhance_Multi3Hate_new/Internvl_new2/ablation" \