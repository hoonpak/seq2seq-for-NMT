cd "/hdd1/user19/bag/4.RNN/src"
CUDA_VISIBLE_DEVICES=0 python train.py --no-reverse --no-dropout --no-input_feeding --sensitive --attn no --align no --name np_v4_base_sensitive --device cuda:0
CUDA_VISIBLE_DEVICES=0 python test.py --no-reverse --no-dropout --no-input_feeding --sensitive --attn no --align no --name np_v4_base_sensitive --device cuda:0
