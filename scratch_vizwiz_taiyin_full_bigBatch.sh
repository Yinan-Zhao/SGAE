python train_mem.py --id log_scratch_taiyin_full_bigBatch --caption_model lstm_mem4 --input_json data/vizwiztalk_taiyin_full.json --input_fc_dir /home/yz9244/AoANet/data/vizwizbu_taiyin_full_fc --input_att_dir /home/yz9244/AoANet/data/vizwizbu_taiyin_full_att --input_box_dir /home/yz9244/AoANet/data/vizwizbu_taiyin_full_box --input_ssg_dir data/vizwiz_taiyin_full_spice_sg2 --input_label_h5 data/vizwiztalk_taiyin_full_label.h5 --sg_dict_path data/vizwiz_taiyin_full_spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 1 --scheduled_sampling_start 0 --checkpoint_path log_scratch_taiyin_full_bigBatch --save_checkpoint_every 544 --val_images_use 5000 --max_epochs 5 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 3 --train_split train --memory_size 10000 --memory_index c --step2_train_after 0 --step3_train_after 0 --use_rela 0 --gpu 0 2>&1 | tee output/output_scratch_taiyin_full_bigBatch.log