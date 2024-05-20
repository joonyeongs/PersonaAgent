export CUDA_VISIBLE_DEVICES=0   ### 이건 우리 전용 GPU 번호 넣으면 됨

LOGFILE="script_execution.log"

'''
echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/infp_cleaned/checkpoint-240 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/infp \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/infp \
--mbti_trait 'INP'\

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/low_lr/infp_cleaned/checkpoint-240 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/infp_low_lr \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/infp_low_lr \
--mbti_trait 'IP'\

pid2=$!
wait $pid2

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/infj_cleaned/checkpoint-111 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/infj \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/infj \
--mbti_trait 'IJ'\

pid3=$!
wait $pid3

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/low_lr/infj_cleaned/checkpoint-111 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/infj_low_lr \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/infj_low_lr \
--mbti_trait 'IN'\

pid4=$!
wait $pid4
'''
echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/entj_cleaned/checkpoint-249 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/entj \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/entj \
--mbti_trait 'ENT'\

pid5=$!
wait $pid5

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/low_lr/entj_cleaned/checkpoint-249 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/entj_low_lr \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/entj_low_lr \
--mbti_trait 'ENT'\

pid6=$!
wait $pid6

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/low_lr/estj_cleaned/checkpoint-117 \
--dataset_dir '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/estj_low_lr \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/estj_low_lr \
--mbti_trait 'S'\

pid7=$!
wait $pid7

echo "Starting new script at $(date)" >> $LOGFILE
python trainer.py \
--model_path /home/vqa/model-weights/llama3/estj_cleaned/checkpoint-117 \
--dataset_dir  '' \
--save_path /home/vqa/model-weights/llama3/machine_mindset/estj \
--output_dir /home/vqa/model-weights/llama3/machine_mindset/estj \
--mbti_trait 'S'\


pid8=$!
wait $pid8

echo "sixth script finished at $(date)" >> $LOGFILE