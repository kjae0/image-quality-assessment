import torch
import argparse
import pickle
from argparse import Namespace

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from models.ensemble_captioning_model import EsembleCaptioningModel
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description

import os
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
    
#  ****************************************************************
#  ****************** ENSEMBLE MODEL CHECKPOINTS ******************
#  *** checkpoint_2023-10-01-20:24:05_epoch0it7113bs4_rf_.pth  ****
#  *** checkpoint_2023-10-01-21:24:05_epoch0it10693bs4_rf_.pth ****
#  *** checkpoint_2023-10-02-01:16:58_epoch1it12296bs4_rf_.pth ****
# *****************************************************************


class ImageDataset(Dataset):
        def __init__(self, image_dir):
            self.image_dir = image_dir 
            self.transform = transforms.Compose([
                transforms.Resize((384,384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
        def __getitem__(self, index):
            img = self.transform(Image.open(self.image_dir[index]).convert("RGB"))
            return img
        
        def __len__(self):
            return len(self.image_dir)

if __name__ == "__main__":
    # set test image folder path
    root_dir = "/data/jaeyeong/dacon/Image_Quality_Assessment/data/test/"
    # root_dir = None
    
    # set model_path
    model_path = [
        '/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-02-01:16:58_epoch1it12296bs4_rf_.pth',
        '/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-20:24:05_epoch0it7113bs4_rf_.pth',
        '/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/github_ignore_material/saves/checkpoint_2023-10-01-21:24:05_epoch0it10693bs4_rf_.pth'
    ]
    # model_path = None
    
    img_name = list(os.listdir(root_dir))
    img_name.sort()
    
    image_files = [os.path.join(root_dir, i) for i in img_name]

    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=74)
    parser.add_argument('--load_path', type=str, default=model_path[0])
    parser.add_argument('--image_paths', type=str,
                        default=image_files,
                        nargs='+')
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           dropout=0.0,
                           drop_args=drop_args)

    with open('/data/jaeyeong/dacon/Image_Quality_Assessment/exv2/ExpansionNet_v2/inference_materials.pkl', 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    print("Dictionary loaded ...")

    img_size = 384
    
    dset = ImageDataset(image_files)
    dl = DataLoader(dset,
                    batch_size=8,
                    shuffle=False,
                    drop_last=False,
                    num_workers=16)
    
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=args.max_seq_len, drop_args=model_args.drop_args,
                                rank='cuda:0')
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda:0')
    model.eval()

    model2 = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=args.max_seq_len, drop_args=model_args.drop_args,
                                rank='cuda:0')
    checkpoint = torch.load(model_path[1])
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2 = model2.to('cuda:0')
    model2.eval()


    model3 = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                    swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                    swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                    swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                    swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                    swin_use_checkpoint=False,
                                    final_swin_dim=1536,

                                    d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                    N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                    num_exp_enc_list=[32, 64, 128, 256, 512],
                                    num_exp_dec=16,
                                    output_word2idx=coco_tokens['word2idx_dict'],
                                    output_idx2word=coco_tokens['idx2word_list'],
                                    max_seq_len=args.max_seq_len, drop_args=model_args.drop_args,
                                    rank='cuda:0')
    checkpoint = torch.load(model_path[2])
    model3.load_state_dict(checkpoint['model_state_dict'])
    model3 = model3.to('cuda:0')
    model3.eval()

    ensemble = EsembleCaptioningModel([model, model2, model3], 'cuda:0')
    ensemble.eval()
    model = ensemble
    print("Model loaded ...")
    
    predictions = []
    print("Generating captions ...\n")
    for i, x in tqdm(enumerate(dl), total=len(dl), ncols=60):
        image = x.to('cuda:0')
        beam_search_kwargs = {'beam_size': args.beam_size,
                              'beam_max_seq_len': args.max_seq_len,
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': sos_idx,
                              'eos_idx': eos_idx}
        with torch.no_grad():
            pred, _ = model(enc_x=image,
                            enc_x_num_pads=[0 for _ in range(len(x))],
                            mode='beam_search', **beam_search_kwargs)
        
        for p in pred:
            p_ = tokens2description(p[0], coco_tokens['idx2word_list'], sos_idx, eos_idx)
            predictions.append(p_)
            
        if (i+1) % 100 == 0:
        # if True:
            # print(path + ' \n\tDescription: ' + pred + '\n')
            try:
                df = pd.DataFrame()
                df['img_name'] = pd.Series(img_name[:len(predictions)])
                df['comments'] = pd.Series(predictions)
                df.to_csv("./comment_prediction.csv", index=False)
            except:
                pass

    df = pd.DataFrame()
    df['img_name'] = pd.Series(img_name)
    df['comments'] = pd.Series(predictions)
    df.to_csv("./comment_prediction.csv", index=False)

    print("Closed.")
