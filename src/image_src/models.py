from torchvision.models import vit_h_14, ViT_H_14_Weights
from swin_models import SwinTransformerV2
from nfnets import pretrained_nfnet

import torch

def build_model(model_name, nf_weight_dir=None):
    if model_name == "ViTH":
        return ViTHuge()
    elif model_name == "Swin":
        return SwinTransformerV2()
    elif model_name == "NFNetF5":
        return NFNetF5(nf_weight_dir)
    else:
        raise NotImplementedError


class NFNetF5(torch.nn.Module):
    def __init__(self, model_dir):
        super(NFNetF5, self).__init__()
        self.model = pretrained_nfnet(model_dir)
        # self.model = pretrained_nfnet("/data/jaeyeong/dacon/Image_Quality_Assessment/IQA/nfnet_pretrained/f5_weight.npz")
    
    def forward(self, x):
        out = self.model.stem(x)
        out = self.model.body(out)
        pool = torch.mean(out, dim=(2,3))
        
        return pool
    

class ViTHuge(torch.nn.Module):
    def __init__(self):
        super(ViTHuge, self).__init__()
        self.model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
    
    def encoder_forward(self, input):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.model.encoder.pos_embedding
        input = self.model.encoder(input)
        
        for layer in self.model.encoder.layers.children():
            input = layer(input)
            
        return self.model.encoder.ln(input)
    
    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # x = self.model.encoder(x)
        x = self.encoder_forward(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x