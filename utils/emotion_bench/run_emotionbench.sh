#bin/bash

python run_emotionbench.py \
  --model gpt-3.5-turbo \
  --questionnaire PANAS \
  --emotion ALL \
  --select-count 999 \
  --default-shuffle-count 0 \
  --emotion-shuffle-count 0 \
  --test-count 1