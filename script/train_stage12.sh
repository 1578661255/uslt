output_dir=out/stage1_pretraining

# deepspeed --include localhost:0 --master_port 29511 pre_training.py \
#    --batch-size 16 \
#    --gradient-accumulation-steps 8 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --quick_break 2048 \
#    --output_dir $output_dir \
#    --dataset CSL_News
# output_dir=out/stage1_pretraining

python pre_training.py \
   --batch-size 2 \
   --gradient-accumulation-steps 1 \
   --epochs 1 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 50 \
   --output_dir $output_dir \
   --dataset CSL_News
