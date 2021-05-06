CUDA_VISIBLE_DEVICES=0 python train.py \
 --task_name test\
 --warm_up_step 100\
 --report_step 1000\
 --eval_step 3000\
 --n_epoch 1\
 --batch_size 32\
 --init_lr 5e-6\
 --embed_dim 64\
 --hidden_size 64\

