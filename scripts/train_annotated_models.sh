export CUDA_VISIBLE_DEVICES=0  ### 이건 우리 전용 GPU 번호 넣으면 됨

LOGFILE="script_execution.log"


echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/infp/INFP_annotated_full.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/infp/full \
--epoch 5 \
--mbti INFP \
--model_path /home/vqa/model-weights/llama3/annotated/infp/ablation/checkpoint-33 \
#--mbti_trait \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/infj/pair_data_final.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/infj/full \
--epoch 5 \
--mbti INFJ \
#--model_path /home/vqa/model-weights/llama3/annotated/infp/ablation/checkpoint-33 \

pid2=$!
wait $pid2

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/entj/entj_pair_data.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/entj/full \
--epoch 5 \
--mbti ENTJ \
#--model_path /home/vqa/model-weights/llama3/annotated/infp/ablation/checkpoint-33 \

pid3=$!
wait $pid3


echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/estj/estj_pair_data.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/estj/full \
--epoch 5 \
--mbti ESTJ \
#--model_path /home/vqa/model-weights/llama3/annotated/infp/ablation/checkpoint-33 \

pid4=$!
wait $pid4

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/estj/estj_gpt_augmented.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/estj/ablation \
--epoch 10 \
--mbti ESTJ \
#--model_path /home/vqa/model-weights/llama3/annotated/infp/ablation/checkpoint-33 \


