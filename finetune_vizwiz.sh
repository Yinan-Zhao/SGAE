python finetune_mem.py --id log_finetune --caption_model lstm_mem4 --input_json data/vizwiztalk_withCOCO.json --input_fc_dir /home/yz9244/AoANet/data/vizwizbu_fc --input_att_dir /home/yz9244/AoANet/data/vizwizbu_att --input_ssg_dir data/vizwiz_withCOCO_spice_sg2 --input_label_h5 data/vizwiztalk_withCOCO_label.h5 --sg_dict_path data/vizwiz_withCOCO_spice_sg_dict2.npz --memory_cell_path /home/yz9244/SGAE/data/caption_data/memory_cellid640075.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 5 --scheduled_sampling_start 0 --checkpoint_path log_finetune --save_checkpoint_every 5000 --val_images_use 5000 --max_epochs 80 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 40 --train_split train --memory_size 10000 --memory_index c --step2_train_after 0 --step3_train_after 0 --use_rela 0 --gpu 2