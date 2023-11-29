MODEL_CONFIG=TODO

DATA_PATH=TODO

OUTPUT_DIR=TODO

BATCH_SIZE=8
GRAD_ACCU=4
LR=2e-7

EVAL_STEP=100




deepspeed train_gptj_summarize.py \
    --model_config $MODEL_CONFIG \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation $GRAD_ACCU \
    --lr $LR \
    --eval_step $EVAL_STEP
