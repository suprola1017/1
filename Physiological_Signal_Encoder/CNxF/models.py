import torch
from torch import nn
import torch.nn.functional as F
import math

import numpy as np

from Physiological_Signal_Encoder.convNeXt.convNext import ConvNeXt
from Physiological_Signal_Encoder.convNeXt.convNext import convnext_base as convnext

class MultiscaleConvolutionalNetwork(nn.Module):
    def __init__(self, params):
        super(MultiscaleConvolutionalNetwork, self).__init__()
        self.channels = params.m1_len+params.m2_len+params.m3_len+params.m4_len
        
        # Convolutional layers with different kernel sizes
        self.conv1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels)
        self.conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=5, padding=2, groups=self.channels)
        self.conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=7, padding=3, groups=self.channels)
        self.conv4 = nn.Conv1d(self.channels, self.channels, kernel_size=9, padding=4, groups=self.channels)
        
        
        # Batch normalization and activation function
        self.batch_norm = nn.BatchNorm1d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling layer
        self.drop = nn.Dropout(0.35)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fconv = nn.Conv1d(self.channels * 4, self.channels, kernel_size=1)

    def forward(self, x):
        # Apply convolutional layers with different kernel sizes
        x1 = self.drop(self.relu(self.batch_norm(self.conv1(x))))
        x2 = self.drop(self.relu(self.batch_norm(self.conv2(x))))
        x3 = self.drop(self.relu(self.batch_norm(self.conv3(x))))
        x4 = self.drop(self.relu(self.batch_norm(self.conv4(x))))
        # x5 = self.drop(self.relu(self.batch_norm(self.conv5(x))))
        # x6 = self.drop(self.relu(self.batch_norm(self.conv6(x))))
        # x7 = self.drop(self.relu(self.batch_norm(self.conv7(x))))
        # x8 = self.drop(self.relu(self.batch_norm(self.conv8(x))))
        # x9 = self.drop(self.relu(self.batch_norm(self.conv9(x))))
        
        
        # Concatenate features from different scales
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.fconv(x)
        
        return x
    
    
class DepthwiseSeparable(nn.Module):
    
    def __init__(self, channel):
        super(DepthwiseSeparable, self).__init__()
        self.channel = channel
        
        self.dsc = nn.Sequential(
            # 深度卷积层
            nn.Conv1d(channel, channel, kernel_size=3, padding=1, groups=channel),  # groups 参数设置为通道数，即4
            nn.BatchNorm1d(channel),
            nn.Dropout(0.35),
            nn.ReLU(inplace=True),
            
            # 逐点卷积层
            nn.Conv1d(channel, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.Dropout(0.35),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x = self.dsc(x)
        return x
        
    

class CNXFModel(nn.Module):
    
    def __init__(self, params):
        super(CNXFModel, self).__init__()
        self.dataset = params.dataset
        self.orig_d_m1, self.orig_d_m2, self.orig_d_m3,self.orig_d_m4  = params.orig_d_m1, params.orig_d_m2, params.orig_d_m3,params.orig_d_m4
        self.m1_len, self.m2_len, self.m3_len, self.m4_len = params.m1_len, params.m2_len, params.m3_len, params.m4_len
        self.mod_num = 4
        if self.dataset == 'wesad':
            self.d_m = 14
        else:
            self.d_m = 32
        self.multi_shape = [32, 32]
        self.output_dim = params.output_dim
        self.layer_scale_init_value = params.layer_scale_init_value
        self.head_init_scale = params.head_init_scale
        self.channels = params.m1_len+params.m2_len+params.m3_len+params.m4_len
        
        # 1. Temporal convolutional layers
        self.proj_m1_hz = nn.Conv1d(self.orig_d_m1, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m2_hz = nn.Conv1d(self.orig_d_m2, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m3_hz = nn.Conv1d(self.orig_d_m3, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m4_hz = nn.Conv1d(self.orig_d_m4, self.d_m, kernel_size=1, padding=0, bias=False)
        
        self.proj_m1_len = nn.Conv1d(self.m1_len, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m2_len = nn.Conv1d(self.m2_len, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m3_len = nn.Conv1d(self.m3_len, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m4_len = nn.Conv1d(self.m4_len, self.d_m, kernel_size=1, padding=0, bias=False)
 
        # 2. ConvNext
        self.trans_m1_all = self.get_network(self_type='m1')
        self.trans_m2_all = self.get_network(self_type='m2')
        self.trans_m3_all = self.get_network(self_type='m3')
        self.trans_m4_all = self.get_network(self_type='m4') 
        
        #3. dsc network for sigle signal
        self.dsc_1 = DepthwiseSeparable(self.m1_len)
        self.dsc_2 = DepthwiseSeparable(self.m2_len)
        self.dsc_3 = DepthwiseSeparable(self.m3_len)
        self.dsc_4 = DepthwiseSeparable(self.m4_len)
        
        #4. msc network for all signal
        self.msc = MultiscaleConvolutionalNetwork(params)
    
        
        #4. ConvNexX Mult
        self.cnxt = self.get_network(self_type="mult")
        
        #4. judge        
        self.final_conv = nn.Conv1d(self.multi_shape[-1], 1, kernel_size=1, padding=0, bias=False)
        self.out_layer = nn.Linear(self.multi_shape[-1], self.output_dim)
        
        # self.lstm = nn.LSTM(self.d_m, self.d_m * 2, num_layers=5, batch_first=True)
        # self.fc = nn.Linear(self.d_m * 2, self.output_dim)
         
    
    def get_network(self, self_type=''):
        
        if self_type in ['m1','m2','m3','m4','mult']:
            return convnext(
                self_type = self_type,
                multi_shape = self.multi_shape,
                layer_scale_init_value = self.layer_scale_init_value,
                head_init_scale = self.head_init_scale
            )
        else:
            return ConvNeXt(
                self_type = self_type,
                multi_shape = self.multi_shape,
                layer_scale_init_value = self.layer_scale_init_value,
                head_init_scale = self.head_init_scale,
                output_dim = self.output_dim
            )
        
            
    def forward(self, m1, m2, m3, m4):
        
        """
        #m1: torch.Size([batch_size, 32, 128])
        #m2: torch.Size([batch_size, 2, 128])
        #m3: torch.Size([batch_size, 2, 128])
        #m4: torch.Size([batch_size, 1, 128]) 
        
        """
            
        m_1 = m1.transpose(1, 2) #torch.Size([batch_size, 128, 32])  torch.Size([batch_size, 14, 50])
        m_2 = m2.transpose(1, 2) #torch.Size([batch_size, 128, 2])   torch.Size([batch_size, 4, 16])
        m_3 = m3.transpose(1, 2) #torch.Size([batch_size, 128, 2])   torch.Size([batch_size, 14, 50])
        m_4 = m4.transpose(1, 2) #torch.Size([batch_size, 128, 1])   torch.Size([batch_size, 4, 1])
        
        #预处理
        proj_x_m1_hz = self.proj_m1_hz(m_1) #torch.Size([batch_size, 32, 32])   torch.Size([batch_size, 14, 50])
        proj_x_m2_hz = self.proj_m2_hz(m_2) #torch.Size([batch_size, 32, 2])    torch.Size([batch_size, 14, 16])
        proj_x_m3_hz = self.proj_m3_hz(m_3) #torch.Size([batch_size, 32, 2])    torch.Size([batch_size, 14, 50])
        proj_x_m4_hz = self.proj_m4_hz(m_4) #torch.Size([batch_size, 32, 1])    torch.Size([batch_size, 14, 1])
        
        #DSC for one signal
        dsc_x_m1 = self.dsc_1(proj_x_m1_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 32])  torch.Size([batch_size, 50, 14])
        dsc_x_m2 = self.dsc_2(proj_x_m2_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 2])   torch.Size([batch_size, 16, 14])
        dsc_x_m3 = self.dsc_3(proj_x_m3_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 2])   torch.Size([batch_size, 50, 14])
        dsc_x_m4 = self.dsc_4(proj_x_m4_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 1])   torch.Size([batch_size, 1, 14])
        
        mul_mod_dsc = torch.cat((dsc_x_m1, dsc_x_m2, dsc_x_m3, dsc_x_m4), dim=1) #torch.Size([batch_size, 37, 32])  torch.Size([batch_size, 117, 14])
    
        #MSC for all signal
        proj_mul_mod = torch.cat((proj_x_m1_hz.permute(0, 2, 1), proj_x_m2_hz.permute(0, 2, 1), proj_x_m3_hz.permute(0, 2, 1), proj_x_m4_hz.permute(0, 2, 1)), dim=1) #torch.Size([batch_size, 37, 32])  torch.Size([batch_size, 117, 14]) 
        mul_mod_msc = self.msc(proj_mul_mod) #torch.Size([batch_size, 37, 32])  torch.Size([batch_size, 117, 14])
        
        
        #结合每层融合特征，进行全局融合
        mul_mod_2_sides = torch.cat((mul_mod_msc, mul_mod_dsc), dim=1) #torch.Size([batch_size, 74, 32])    torch.Size([batch_size, 234, 14])    
        if self.dataset == 'wesad': 
            mul_mod_cnxt_att = self.cnxt(mul_mod_2_sides.reshape(-1, 91, 36).unsqueeze(1))  #torch.Size([batch_size, 32, 32])
        else:
            mul_mod_cnxt_att = self.cnxt(mul_mod_2_sides.unsqueeze(1))  #torch.Size([batch_size, 32, 32])
        
        #分类器
        last_hs = self.final_conv(mul_mod_cnxt_att).squeeze(1) #torch.Size([batch_size, 32])
        output = self.out_layer(last_hs)
        
        # last_hs, _ = self.lstm(mul_mod_cnxt_att) #torch.Size([batch_size, 32, 64])
        # lstm_out_last = last_hs[:, -1, :] ## Take the output of the last time step #torch.Size([batch_size, 64])

        # # Classification
        # output = self.fc(lstm_out_last)
        
        
        
       
        
        #分类
        # output = self.convnext_output(mul_mod_msc.unsqueeze(1)) #torch.Size([batch_size, 1])
        
        # proj_x_m1_len = self.proj_m1_len(proj_x_m1_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 32])
        # proj_x_m2_len = self.proj_m2_len(proj_x_m2_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 2])
        # proj_x_m3_len = self.proj_m3_len(proj_x_m3_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 2])
        # proj_x_m4_len = self.proj_m4_len(proj_x_m4_hz.permute(0, 2, 1)) #torch.Size([batch_size, 32, 1])

      
        # proj_x_m1 = proj_x_m1_hz.unsqueeze(1) #torch.Size([batch_size, 1, 32, 32])
        # proj_x_m2 = proj_x_m2_hz.unsqueeze(1) #torch.Size([batch_size, 1, 32, 32])
        # proj_x_m3 = proj_x_m3_hz.unsqueeze(1) #torch.Size([batch_size, 1, 32, 32])
        # proj_x_m4 = proj_x_m4_hz.unsqueeze(1) #torch.Size([batch_size, 1, 32, 32])

        # #各自进入网络
        # m1_with_all = self.trans_m1_all(proj_x_m1)  #torch.size([batch_size, 32, 24])
        # m2_with_all = self.trans_m2_all(proj_x_m2)  #torch.size([batch_size, 32, 24])
        # m3_with_all = self.trans_m3_all(proj_x_m3)  #torch.size([batch_size, 32, 24])
        # m4_with_all = self.trans_m4_all(proj_x_m4)  #torch.size([batch_size, 32, 24])
        
        # #融合模态
        # mul_mod = torch.cat((m1_with_all.unsqueeze(1), m2_with_all.unsqueeze(1), m3_with_all.unsqueeze(1), m4_with_all.unsqueeze(1)), dim=1) #torch.Size([batch_size, 4, 32, 24])
        # mul_mod_final = self.dsc(mul_mod).squeeze(1) #torch.Size([batch_size, 32, 24])
        
        
        # #对融合的模态进行训练
        # fc = self.final_conv(mul_mod_final).squeeze(1) #torch.Size([batch_size, 24])
        # output = self.out_layer(fc) #torch.Size([batch_size, 1])
        
        # mul_model_process = mul_model.unsqueeze(1) #torch.Size([batch_size, 1, 64, 48])
        # output = self.out_layer(mul_model_process) #torch.Size([batch_size, 1])

        
        
        #husformerd 的 output：tensor([[ 3.0447e-01],[-2.6315e-02],[ 1.8127e-01], [ 4.1449e-01],[ 5.3485e-01],
        
        
        return output, mul_mod_cnxt_att