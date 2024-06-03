
echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/infp/INFP_annotated_all.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/infp/full/mixed \
--epoch 3 \
--mbti INFP \



#--mbti_trait \


echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/infj/pair_data_chatgpt_augmented.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/infj/ablation \
--epoch 10 \
--mbti INFJ \
#--model_path /home/vqa/model-weights/llama3/annotated/infj/ablation/checkpoint-56 \


echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/infj/pair_data_final.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/infj/full/ \
--epoch 3 \
--mbti INFJ \


echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--dataset_dir /home/vqa/bell_hoon/Agents/PersonaAgent_final/data/annotated_data/estj/estj_pair_data.json \
--beta 0.2  \
--output_dir /home/vqa/model-weights/llama3/annotated/estj/full/ \
--epoch 3 \
--mbti ESTJ \






