# from turtle import forward
from pyparsing import Forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Layer_norm, ConvNeXtBlock
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1) 
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1) 
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1) 
        self.gamma = nn.Parameter(torch.zeros(1))  

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B, C, W, H) 
            returns :
                out : self attention value + input feature 
                attention: B, N, N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B, CX(N) 
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) 
        energy =  torch.bmm(proj_query,proj_key) # transpose check     
        attention = self.softmax(energy) # B, (N), (N)     
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B, C, N  

        out = torch.bmm(proj_value,attention.permute(0,2,1) ) 
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x  
        return out


class ConvNeXt(nn.Module):
    """
    Args:
        in_channels (int): Number of input image channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 100
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    
    """

    def __init__(self, in_channels=1, num_classes=4, depths=[3, 3, 9, 3], 
                       dims=[96, 192, 384, 768], drop_path_rate=0.2, 
                       layer_scale_init_value=1e-6, head_init_scale=1.,
                       self_type = 'judge', multi_shape = [],
                       output_dim = 1
                 ):

        super(ConvNeXt, self).__init__()

        self.self_type = self_type
        self.multi_shape = multi_shape
        self.dims = dims
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            Layer_norm(dims[0], eps=1e-6, input_format="Channel_First")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                Layer_norm(dims[i], eps=1e-6, input_format="Channel_First"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.attentions = nn.ModuleList()
        dp_rates = [x for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(in_channel = dims[i], depth_rate=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])],
                Layer_norm(dims[i], eps=1e-6, input_format="Channel_First"),  
            )
            attention = Self_Attn(dims[i], 'relu')
            
            self.attentions.append(attention)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], output_dim)

        self.apply(self.init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def init_weights(self, m):

        if isinstance(m , (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias,0)

    def forward_stages(self, x):
        for i in range(4):
            #i:0:torch.Size([16, 128, 18, 8]);torch.Size([16, 128, 22, 9]) i:1:torch.Size([16, 256, 9, 4]) ; torch.Size([16, 256, 11, 4])
            #i:2:torch.Size([16, 512, 4, 2]); torch.Size([16, 512, 5, 2]) i:3:torch.Size([16, 1024, 2, 1])
            x = self.downsample_layers[i](x) 
            x = self.stages[i](x)
            #attention layer
            x = self.attentions[i](x)
        
        
        
        if self.self_type == 'judge':
            return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)  ——————————》 N（Batch Size） C（Channels） H（Height） W（Width）
        else:
            x = self.norm(x.mean([-2, -1])) #torch.Size([16, 1024])
            return x.view(-1, self.multi_shape[0], self.multi_shape[1])
        
        
    def forward(self, x, y=None):
        #begin:torch.Size([16, 1, 74, 32]); end:torch.Size([1024, 768])
        #begin:torch.Size([16, 1, 91, 36]); end:torch.Size([1024, 768])
        x = self.forward_stages(x)
        
        if self.self_type == "judge":
            x = self.head(x) 
        
        return x

# Pretrained models for testing

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

