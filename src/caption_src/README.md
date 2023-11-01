코드 실행 순서입니다.
(해당 논문에서 활용한 방법을 사용했습니다. https://github.com/jchenghu/ExpansionNet_v2)

**
- 로그는 .txt 형태로 저장됩니다.
- ckpt들이 저장 될 github_ignore_materia/raw_data, github_ignore_materia/saves는 미리 만들어져야 합니다.
- 학습 진행 시, end-to-end Step에서는 gpu 메모리 이슈로 evalutaion을 생략하였습니다. train.py의 line 165, 166의 주석 처리를 풀면 매 epoch 마다 evaluation이 가능합니다.

0. Before start
- 학습에 활용될 comment 파일을 coco dataset format으로 변형시켜 저장해둡니다.
- build_train_json.py를 실행시키면 됩니다.
- *** train.json 생성 과정에서 set 자료형 사용으로 인해 train, val, test split이 실험 환경과 달라지기 때문에, 재현을 위해서는 첨부된 train.json을 활용해야 합니다.

1. Data generation for partial training (Cross entropy loss)
- backbone은 SwinTransformer로, pretrained model을 활용합니다. 
- 논문 저자들이 제공하는 pretrained weight 중에서 xe_model을 활용했습니다. (pretrained from coco dataset)
- ~/caption_features/features.hdf5 가 주어진 train dataset에서 추출한 feature입니다.
- model path에는 expansionv2_pretrained_weights/xe_model.pth를 넣어주면 됩니다.

<!-- Command -->
python data_generator.py \
    --save_model_path (model path) \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --captions_path ./github_ignore_material/raw_data/train.json &> data_generation.txt &

<!-- Example -->
python data_generator.py \
    --save_model_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/pretrained/xe_model.pth \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --captions_path ./github_ignore_material/raw_data/train.json &> data_generation.txt &

2. Partial training (Cross entropy loss)
- step 1에서 추출한 feature로 backbone을 제외한 부분을 학습시킵니다.
- 논문 저자들이 제공하는 pretrained weight 중에서 xe_model을 활용했습니다. (pretrained from coco dataset)
- train.py를 실행시키면 됩니다.
- model path에는 expansionv2_pretrained_weights/xe_model.pth를 넣어주면 됩니다.

<!-- command -->
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

<!-- example -->
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

3. End to end training (Cross entropy loss)
- model 전체를 학습시킵니다.
- step 2에서 저장된 마지막 epoch에서 저장된 model weight를 활용합니다
- train.py를 실행시키면 됩니다.
- backbone path에는 data generation에서 사용된 모델, model path에는 이전 스텝에서 저장된 마지막 epoch 모델 weight path를 넣어주면 됩니다.

<!-- command -->
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533   --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 8 --num_accum 6 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --partial_load True \
    --backbone_save_path (backbone path) \
    --body_save_path (model path) \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 5 &> end_to_end_celoss.txt &

<!-- example -->
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533   --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 8 --num_accum 6 --num_gpus 4 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --partial_load True \
    --backbone_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/pretrained/xe_model.pth \
    --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth \
    --print_every_iter 100 --eval_every_iter 999999 \
    --reinforce False --num_epochs 5 &> end_to_end_celoss.txt &

4. Data generation for partial training (CIDEr optimization)
- backbone은 SwinTransformer로, pretrained model을 활용합니다. 
- pretrained model weight는 step 3에서 나온 모델입니다.
- ~/caption_features/features_cider.hdf5 가 주어진 train dataset에서 추출한 feature입니다.
- model path에는 이전 스텝에서 저장된 마지막 epoch 모델 weight path를 넣어주면 됩니다.

<!-- command -->
python data_generator.py \
    --save_model_path (model path) \
    --output_path ./github_ignore_material/raw_data/features_cider.hdf5 \
    --captions_path ./train.json &> data_generation_cider.txt &

<!-- example -->
python data_generator.py \
    --save_model_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth \
    --output_path ./github_ignore_material/raw_data/features_cider.hdf5 \
    --captions_path ./train.json &> data_generation_cider.txt &

5. Partial training (CIDEr optimization)
- step 3에서 추출한 feature로 backbone을 제외한 부분을 학습시킵니다.
- train.py를 실행시키면 됩니다.
- model path에는 이전 스텝에서 저장된 마지막 epoch 모델 weight path를 넣어주면 됩니다.
- feature path에는 이전 스텝에서 추출한 feature의 경로를 넣어주면 됩니다.

<!-- command -->
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

<!-- example -->
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


6. End to end training (CIDEr optimization)
- model 전체를 학습시킵니다.
- step 5에서 저장된 마지막 epoch에서 저장된 model weight를 활용합니다
- train.py를 실행시키면 됩니다.
- backbone path, model path에는 이전 스텝에서 저장된 마지막 epoch 모델 weight path를 넣어주면 됩니다.
- save path는 저장 경로입니다.

<!-- command -->
python train.py --N_enc 3 --N_dec 3     
 --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      \
 --warmup 100 --anneal_coeff 0.7 --lr 1e-5 --enc_drop 0.1     --dec_drop 0.1 --enc_input_drop 0.1 \
 --dec_input_drop 0.1 --drop_other 0.1      --batch_size 4 --num_accum 12 --num_gpus 4 \
 --ddp_sync_port 11317 --eval_beam_sizes [5]      --save_path (save_path) \
 --save_every_minutes 60 --how_many_checkpoints 10      --is_end_to_end True --partial_load True     \
 --backbone_save_path (backbone_path)     \
 --body_save_path (model path)     \
 --print_every_iter 300 --eval_every_iter 999999     --reinforce True --num_epochs 2 &> end_to_end_cider.txt &>

<!-- example -->
python train.py --N_enc 3 --N_dec 3     
 --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      \
 --warmup 100 --anneal_coeff 0.7 --lr 1e-5 --enc_drop 0.1     --dec_drop 0.1 --enc_input_drop 0.1 \
 --dec_input_drop 0.1 --drop_other 0.1      --batch_size 4 --num_accum 12 --num_gpus 4 \
 --ddp_sync_port 11317 --eval_beam_sizes [5]      --save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/ \
 --save_every_minutes 60 --how_many_checkpoints 10      --is_end_to_end True --partial_load True     \
 --backbone_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-06:34:30_epoch4it8388bs8_xe_.pth     \
 --body_save_path /data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-17:31:44_epoch4it4096bs12_rf_.pth     \
 --print_every_iter 300 --eval_every_iter 999999     --reinforce True --num_epochs 2 &> end_to_end_cider.txt &

7. Inference
- inference.py를 실행하면 됩니다.

8. Submission
- mos_prediction.pkl, comment_prediction.csv를 생성한 후, make_submission.py를 실행하면 됩니다.
