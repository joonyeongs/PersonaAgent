#!/bin/bash

# Running the Python script for each MBTI type
echo "Running for MBTI type: none"
python beavertails.py \
    --mbti "none" \
    --model-dir "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input-dir "/home/vqa/data/dataset/beavertails/test.jsonl" \
    --output-dir "/home/vqa/data/outputs/beavertails"


echo "Running for MBTI type: INFP"
python beavertails.py \
    --mbti "INFP" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infp/full/mixed/checkpoint-123" \
    --input-dir "/home/vqa/data/dataset/beavertails/test.jsonl" \
    --output-dir "/home/vqa/data/outputs/beavertails"


echo "Running for MBTI type: ENTJ"
python beavertails.py \
    --mbti "ENTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/entj/full/checkpoint-142" \
    --input-dir "/home/vqa/data/dataset/beavertails/test.jsonl" \
    --output-dir "/home/vqa/data/outputs/beavertails"

echo "Running for MBTI type: INFJ"
python beavertails.py \
    --mbti "INFJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infj/ablation/checkpoint-56" \
    --input-dir "/home/vqa/data/dataset/beavertails/test.jsonl" \
    --output-dir "/home/vqa/data/outputs/beavertails"

echo "Running for MBTI type: ESTJ"
python beavertails.py \
    --mbti "ESTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/estj/full/checkpoint-78" \
    --input-dir "/home/vqa/data/dataset/beavertails/test.jsonl" \
    --output-dir "/home/vqa/data/outputs/beavertails"
