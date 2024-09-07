# image-quality-assessment
Private 1st code of AI competition "2023 Samsung AI Challenge : Image Quality Assessment" hosted by Samsung AIT &amp; Dacon 

## Overview
There are two tasks for this competition, image quality assesment and image captioning.

### image quality assessment
1. Extract features from image with pretrained neural network
2. training autogluon model with extracted features.
I extracted features from 4 pretraiend models with 2 version of input. (total 8)
- Models: ViT Huge, Swin Transformer 1K, Swin Transformer 22K, NFNet F5. 
- Inputs: original image, horizontal flipped image.

Final result is average ensemble of them.

### image captioning
I utilized ExpansionNet V2, which outperforms among other arhcitectures. <br>
Check src/caption_src/ README.md for details.

## Get Started
### Feature extractin for image quality assessment
~~~ bash
python ./src/image_src/feature_extraction.py --csv_dir <csv_dir> --root_dir <root_dir> --save_dir <save_dir> --model_name <model_name> --img_size <img_size>
~~~

- ViT Huge - model_name: ViT_H, img_size: 518
- Swin 1k - model_name: Swin, img_size: 384 
- Swin 22k - model_name: Swin, img_size: 192 
- NFNet F5 - model_name: NFNetF5, img_size: 544

flip argument is optional, and every model requries pretrained weight. (state_dict_dir, nf_weight_dir)

For extracted feature files, please don't hesitate to reach out to me via Gmail.

### Run autogluon model
For training,
~~~bash
python ./src/image_src/autogluon_reressor.py --data <data_dir>
~~~

For inference,
~~~bash
python ./src/image_src/autogluon_inference.py
~~~

## Sources
I utilized code for image captioning repository below (ExpansionNet V2).
https://github.com/jchenghu/ExpansionNet_v2
