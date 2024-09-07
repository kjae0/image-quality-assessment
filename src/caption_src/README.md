# Code Execution Workflow
(This approach is adapted from the following paper: [https://github.com/jchenghu/ExpansionNet_v2](https://github.com/jchenghu/ExpansionNet_v2))

### Important Notes:
- Logs will be saved in `.txt` format.
- Make sure to pre-create the directories `github_ignore_materia/raw_data` and `github_ignore_materia/saves` for storing checkpoints.
- For end-to-end steps, evaluation is skipped due to GPU memory constraints. If you wish to enable evaluation during each epoch, uncomment lines 165 and 166 in `train.py`.

---

### 0. Before You Start
- Convert the comment file to the COCO dataset format for training.
- Run `build_train_json.py` to generate the required `train.json`.
- **Important**: The use of set data structures during the creation of `train.json` may result in different train/validation/test splits from those used in the experiment. To reproduce the results accurately, use the provided `train.json`.

---

### 1. Data Generation for Partial Training (Cross-Entropy Loss)
- We use **SwinTransformer** as the backbone and a pretrained model.
- For the pretrained model weights, use the `xe_model` provided by the authors, pretrained on the COCO dataset.
- `~/caption_features/features.hdf5` contains the extracted features from the given training dataset.
- For the `model path`, use `expansionv2_pretrained_weights/xe_model.pth`.

#### Command:
```bash
python data_generator.py \
    --save_model_path (model path) \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --captions_path ./github_ignore_material/raw_data/train.json &> data_generation.txt &
