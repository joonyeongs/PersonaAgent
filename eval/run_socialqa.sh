#!/bin/bash

# Running the Python script for each MBTI type
echo "Running for MBTI type: none"
python socialqa.py \
    --mbti "none" \
    --model-dir "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input-dir "/home/vqa/data/dataset/socialiqa-train-dev/dev.jsonl" \
    --output-dir "/home/vqa/data/outputs/socialqa"
'''
echo "Running for MBTI type: INFP"
python socialqa.py \
    --mbti "INFP" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infp/full/mixed/checkpoint-123" \
    --input-dir "/home/vqa/data/dataset/socialiqa-train-dev/dev.jsonl" \
    --output-dir "/home/vqa/data/outputs/socialqa"


echo "Running for MBTI type: ENTJ"
python socialqa.py \
    --mbti "ENTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/entj/full/checkpoint-142" \
    --input-dir "/home/vqa/data/dataset/socialiqa-train-dev/dev.jsonl" \
    --output-dir "/home/vqa/data/outputs/socialqa"

echo "Running for MBTI type: INFJ"
python socialqa.py \
    --mbti "INFJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infj/ablation/checkpoint-56" \
    --input-dir "/home/vqa/data/dataset/socialiqa-train-dev/dev.jsonl" \
    --output-dir "/home/vqa/data/outputs/socialqa"

echo "Running for MBTI type: ESTJ"
python socialqa.py \
    --mbti "ESTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/estj/full/checkpoint-78" \
    --input-dir "/home/vqa/data/dataset/socialiqa-train-dev/dev.jsonl" \
    --output-dir "/home/vqa/data/outputs/socialqa"
'''