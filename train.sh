num_classes=24
    class_num_per_step=3
    num_workers=0
    memory_size=340
    max_epoches=200
    lam_I=0.5
    lam_C=1.0

cd train

CUDA_VISIBLE_DEVICES=0 nohup python -u train_incremental.py \
                                --num_classes $num_classes \
                                --class_num_per_step $class_num_per_step \
                                --max_epoches $max_epoches \
                                --num_workers $num_workers \
                                --memory_size $memory_size \
                                --instance_contrastive_temperature 0.05 \
                                --class_contrastive_temperature 0.05 \
                                --lam 0.5 \
                                --lam_I $lam_I \
                                --lam_C $lam_C \
                                --lr 1e-3 \
                                --lr_decay False \
                                --milestones 100 \
                                --weight_decay 1e-4 \
                                --train_batch_size 256 \
                                --infer_batch_size 128 \
                                --exemplar_batch_size 128 > nohup-I_$lam_I-C_$lam_C.log 2>&1 &


