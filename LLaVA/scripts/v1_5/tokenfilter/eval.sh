CKPT="llava-v1.5-7b-PatchConcat"

CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/mme.sh $CKPT &
CUDA_VISIBLE_DEVICES=7 . scripts/v1_5/eval/mmbench.sh $CKPT &
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/pope.sh $CKPT &
CUDA_VISIBLE_DEVICES=2 . scripts/v1_5/eval/sqa.sh $CKPT &
CUDA_VISIBLE_DEVICES=3,4 . scripts/v1_5/eval/textvqa.sh $CKPT &
CUDA_VISIBLE_DEVICES=5,6 . scripts/v1_5/eval/vqav2.sh $CKPT &

