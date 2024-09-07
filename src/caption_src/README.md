### Code Execution Steps  
(The method applied here is based on the following paper: https://github.com/jchenghu/ExpansionNet_v2)

**Notes**:
- Logs will be saved in `.txt` format.  
- Directories for checkpoints, `github_ignore_materia/raw_data`, and `github_ignore_materia/saves` should be created beforehand.  
- During training, the evaluation step is skipped for the end-to-end process due to GPU memory issues. To enable evaluation after each epoch, remove the comments from lines 165 and 166 in `train.py`.

### 0. Before Starting
- Convert the comment files to COCO dataset format for training.  
- Execute `build_train_json.py`.  
- **Note**: Due to the use of the `set` data type in generating `train.json`, the `train`, `val`, and `test` splits may differ from the original experiment settings. To ensure reproducibility, use the attached `train.json` file.

---

### 1. Data Generation for Partial Training (Cross Entropy Loss)
- The backbone used is SwinTransformer with a pretrained model.  
- The `xe_model` pretrained weight provided by the paper authors (pretrained on COCO dataset) is used.  
- `~/caption_features/features.hdf5` is the extracted feature from the provided training dataset.  
- Set `expansionv2_pretrained_weights/xe_model.pth` as the model path.

**Command**:
```bash
python data_generator.py \
    --save_model_path (model path) \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --captions_path ./github_ignore_material/raw_data/train.json &> data_generation.txt &
```

### 2. Partial Training (Cross Entropy Loss)
- Train the model excluding the backbone using the features extracted in step 1.
- Use the xe_model pretrained weight provided by the authors.
- Run train.py to begin training.
- Set expansionv2_pretrained_weights/xe_model.pth as the model path.


**Command**:
~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --seed 775533 --optim_type radam --sched_type custom_warmup_anneal  \
    --warmup 300 --lr 5e-4 --anneal_coeff 0.8 --anneal_every_epoch 2 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 48 --num_accum 1 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --body_save_path (model path) \
    --is_end_to_end False --features_path (feature path) --partial_load False \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 15 &> partial_training_celoss.txt &
~~~

Example:
~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --seed 775533 --optim_type radam --sched_type custom_warmup_anneal  \
    --warmup 300 --lr 5e-4 --anneal_coeff 0.8 --anneal_every_epoch 2 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 48 --num_accum 1 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/pretrained/xe_model.pth \
    --is_end_to_end False --features_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/raw_data/features.hdf5 --partial_load False \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 15 &> partial_training_celoss.txt &
~~~

### 3. End-to-End Training (Cross Entropy Loss)
- Train the full model.
- Use the model weights from the last epoch of step 2.
- Run train.py to begin training.
- Set the backbone path to the model used in data generation, and the model path to the weights saved from the last epoch in the previous step.

**Command**:
~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 8 --num_accum 6 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --partial_load True \
    --backbone_save_path (backbone path) \
    --body_save_path (model path) \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 5 &> end_to_end_celoss.txt &
~~~
Example:

~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 8 --num_accum 6 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --partial_load True \
    --backbone_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/pretrained/xe_model.pth \
    --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 5 &> end_to_end_celoss.txt &
~~~

### 4. Data Generation for Partial Training (CIDEr Optimization)
- Use the SwinTransformer backbone with a pretrained model.
- The pretrained model weights should be the ones obtained from step 3.
~/caption_features/features_cider.hdf5 will store the extracted features from the given training dataset.
- Set the model path to the weights saved in the last epoch of the previous step.

**Command**:
~~~bash
python data_generator.py \
    --save_model_path (model path) \
    --output_path ./github_ignore_material/raw_data/features_cider.hdf5 \
    --captions_path ./train.json &> data_generation_cider.txt &
~~~

Example:

~~~bash
python data_generator.py \
    --save_model_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth \
    --output_path ./github_ignore_material/raw_data/features_cider.hdf5 \
    --captions_path ./train.json &> data_generation_cider.txt &
~~~

### 5. Partial Training (CIDEr Optimization)
- Train the model excluding the backbone using the features extracted in step 4.
- Run train.py to start the training process.
- Set the model path to the weights saved in the last epoch of the previous step, and feature path to the extracted features.

**Command**:

~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 300 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 48 --num_accum 1 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 10  \
    --is_end_to_end False --partial_load True \
    --features_path (feature path) \
    --body_save_path (model path) \
    --print_every_iter 300 --eval_every_iter 99999 \
    --reinforce True --num_epochs 15 &> partial_training_cider.txt &
~~~

Example:
~~~bash
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 300 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 12 --num_accum 4 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 10  \
    --is_end_to_end False --partial_load True \
    --features_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/raw_data/features_cider.hdf5 \
    --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-06:34:30_epoch4it8388bs8_xe_.pth \
    --print_every_iter 300 --eval_every_iter 99999 \
    --reinforce True --num_epochs 15 &> partial_training_cider.txt &
6. End-to-End Training (CIDEr Optimization)
Train the entire model.
Use the model weights from the last epoch of step 5.
Run train.py to start training.
Set the backbone path and model path to the weights saved from the last epoch in the previous step.
Command:

bash
python train.py --N_enc 3 --N_dec 3  \
 --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
 --warmup 100 --anneal_coeff 0.7 --lr 1e-5 --enc_drop 0.1 --dec_drop 0.1 --enc_input_drop 0.1 \
 --dec_input_drop 0.1 --drop_other 0.1 --batch_size 4 --num_accum 12 --num_gpus 4 \
 --ddp_sync_port 11317 --eval_beam_sizes [5] --save_path (save_path) \
 --save_every_minutes 60 --how_many_checkpoints 10 --is_end_to_end True --partial_load True \
 --backbone_save_path (backbone path) \
 --body_save_path (model path) \
 --print_every_iter 300 --eval_every_iter 999999 --reinforce True --num_epochs 2 &> end_to_end_cider.txt &
~~~

Example:
~~~bash
python train.py --N_enc 3 --N_dec 3  \
 --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
 --warmup 100 --anneal_coeff 0.7 --lr 1e-5 --enc_drop 0.1 --dec_drop 0.1 --enc_input_drop 0.1 \
 --dec_input_drop 0.1 --drop_other 0.1 --batch_size 4 --num_accum 12 --num_gpus 4 \
 --ddp_sync_port 11317 --eval_beam_sizes [5] --save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/ \
 --save_every_minutes 60 --how_many_checkpoints 10 --is_end_to_end True --partial_load True \
 --backbone_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-06:34:30_epoch4it8388bs8_xe_.pth \
 --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth \
 --print_every_iter 300 --eval_every_iter 999999 --reinforce True --num_epochs 2 &> end_to_end_cider.txt &
~~~

7. Inference
Execute inference.py for model inference.

8. Submission
After generating mos_prediction.pkl and comment_prediction.csv, run make_submission.py for submission.