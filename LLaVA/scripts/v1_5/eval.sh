CKPT="llava-v1.5-7b"

CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/mme.sh $CKPT
CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/mmbench.sh $CKPT
CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/pope.sh $CKPT
CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/sqa.sh $CKPT
CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/textqa.sh $CKPT
CUDA_VISIBLE_DEVICES=3 . scripts/v1_5/eval/vqav2.sh $CKPT
