export CUDA_VISIBLE_DEVICES=0  ### 이건 우리 전용 GPU 번호 넣으면 됨

LOGFILE="script_execution.log"


echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b infp \
--mbti_a none \


pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a infj \
--mbti_b none \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a none \
--mbti_b entj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a estj \
--mbti_b none \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a estj \
--mbti_b infp \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a infj \
--mbti_b estj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a infp \
--mbti_b infj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a entj \
--mbti_b estj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a entj \
--mbti_b infp \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_a infj \
--mbti_b entj \

pid1=$!
wait $pid1


echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b none \
--mbti_a infp \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b infj \
--mbti_a none \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b none \
--mbti_a entj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b estj \
--mbti_a none \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b estj \
--mbti_a infp \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b infj \
--mbti_a estj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b infp \
--mbti_a infj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b entj \
--mbti_a estj \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b entj \
--mbti_a infp \

pid1=$!
wait $pid1

echo "Starting new script at $(date)" >> $LOGFILE
python experiments/sotopia/sotopia_simulation.py \
--mbti_b infj \
--mbti_a entj \

pid1=$!
wait $pid1
