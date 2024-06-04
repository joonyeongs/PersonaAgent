#!/bin/bash

# Running the Python script for each MBTI type
echo "Running for MBTI type: none"
python eqbench.py \
    --mbti "none" \
    --model-dir "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input-dir "/home/vqa/data/dataset/eqbench/eq_bench_v2.json" \
    --output-dir "/home/vqa/data/outputs/eqbench"


echo "Running for MBTI type: INFP"
python eqbench.py \
    --mbti "INFP" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infp/full/mixed/checkpoint-123" \
    --input-dir "/home/vqa/data/dataset/eqbench/eq_bench_v2.json" \
    --output-dir "/home/vqa/data/outputs/eqbench"


echo "Running for MBTI type: ENTJ"
python eqbench.py \
    --mbti "ENTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/entj/full/checkpoint-142" \
    --input-dir "/home/vqa/data/dataset/eqbench/eq_bench_v2.json" \
    --output-dir "/home/vqa/data/outputs/eqbench"

echo "Running for MBTI type: INFJ"
python eqbench.py \
    --mbti "INFJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/infj/ablation/checkpoint-56" \
    --input-dir "/home/vqa/data/dataset/eqbench/eq_bench_v2.json" \
    --output-dir "/home/vqa/data/outputs/eqbench"

echo "Running for MBTI type: ESTJ"
python eqbench.py \
    --mbti "ESTJ" \
    --model-dir "/home/vqa/model-weights/llama3/annotated/estj/full/checkpoint-78" \
    --input-dir "/home/vqa/data/dataset/eqbench/eq_bench_v2.json" \
    --output-dir "/home/vqa/data/outputs/eqbench"
