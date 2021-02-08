#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:36:59 2020

@author: t.yamamoto
"""

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self,Cin,Cout):
        super(Encoder, self).__init__()
        self.Cin = Cin
        self.Cout = Cout
        
        self.conv = nn.Conv2d(self.Cin, self.Cout, 4, 2, 1)
        self.bn=nn.BatchNorm2d(self.Cout)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,Cin,Cout):
        super(Decoder, self).__init__()
        self.Cin = Cin
        self.Cout = Cout
        self.deconv = nn.ConvTranspose2d(2*Cin, Cout, 4,2,1)
        self.bn = nn.BatchNorm2d(self.Cout)
        self.relu = nn.ReLU()
        
    def forward(self,x,skip):
        x = torch.cat([x,skip],dim=1)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class UnetConv2(nn.Module):
    def __init__(self):
        super(UnetConv2, self).__init__()
        self.firstCh = 1
        self.finalCh = 1

        self.kernel_size = 4
        self.stride = 2
        self.pudding = 1

        self.ch = [16,32,64,128,256,512]
        
        self.encoder1=Encoder(self.firstCh, self.ch[0])
        self.encoder2=Encoder(self.ch[0], self.ch[1])
        self.encoder3=Encoder(self.ch[1], self.ch[2])
        self.encoder4=Encoder(self.ch[2], self.ch[3])
        self.encoder5=Encoder(self.ch[3], self.ch[4])
        self.encoder6=Encoder(self.ch[4], self.ch[5])
        
        self.decoder6=Decoder(self.ch[5], self.ch[4])
        self.decoder5=Decoder(self.ch[4], self.ch[3])
        self.decoder4=Decoder(self.ch[3], self.ch[2])
        self.decoder3=Decoder(self.ch[2], self.ch[1])
        self.decoder2=Decoder(self.ch[1], self.ch[0])
        self.decoder1=Decoder(self.ch[0], self.finalCh)
        
    def forward(self,x):
        sh = x.shape
        x = x[:,:512,:] #20 ,512 ,128 ちょっとちっちゃくする
        x = torch.unsqueeze(x,1) #20, 1, 512, 128 #チャンネル次元の追加

        x1 = self.encoder1(x) # 20, 16, 256, 64
        x2 = self.encoder2(x1) #20, 32, 128, 32
        x3 = self.encoder3(x2) #20, 64, 64, 16
        x4 = self.encoder4(x3) #20, 128, 32, 8
        x5 = self.encoder5(x4) #20, 256, 16, 4
        x6 = self.encoder6(x5) #20, 512, 8, 2
        
        #LSTMとか追加するならここ
        y = x6 #20, 512, 8, 2
        
        y = self.decoder6(y, x6)
        y = self.decoder5(y, x5)
        y = self.decoder4(y, x4)
        y = self.decoder3(y, x3)
        y = self.decoder2(y, x2)
        y = self.decoder1(y, x1)
        
        y = torch.sigmoid(y)
        
        y = torch.squeeze(y) # 20, 512, 128
        y0 = torch.zeros(sh)
        y0[:,:512,:] = y

        return y0
        
        
        

class UnetConv(nn.Module):
    def __init__(self): # input channel / output channels
        super(UnetConv, self).__init__()

        self.firstCh = 1
        self.finalCh = 1

        self.kernel_size = 4
        self.stride = 2
        self.pudding = 1

        self.ch = [16,32,64,128,256,512]

        self.conv1 = nn.Conv2d(self.firstCh, self.ch[0], self.kernel_size, self.stride, self.pudding) # first -> 16
        self.conv2 = nn.Conv2d(self.ch[0], self.ch[1], self.kernel_size, self.stride, self.pudding) # 16 -> 32
        self.conv3 = nn.Conv2d(self.ch[1], self.ch[2], self.kernel_size, self.stride, self.pudding) # 32 -> 64
        self.conv4 = nn.Conv2d(self.ch[2], self.ch[3], self.kernel_size, self.stride, self.pudding) # 64 -> 128
        self.conv5 = nn.Conv2d(self.ch[3], self.ch[4], self.kernel_size, self.stride, self.pudding) # 128 -> 256
        self.conv6 = nn.Conv2d(self.ch[4], self.ch[5], self.kernel_size, self.stride, self.pudding) # 256 -> 512

        self.bn1=nn.BatchNorm2d(self.ch[0])
        self.bn2=nn.BatchNorm2d(self.ch[1])
        self.bn3=nn.BatchNorm2d(self.ch[2])
        self.bn4=nn.BatchNorm2d(self.ch[3])
        self.bn5=nn.BatchNorm2d(self.ch[4])
        self.bn6=nn.BatchNorm2d(self.ch[5])

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.deconv6 = nn.ConvTranspose2d(2*self.ch[5], self.ch[4], self.kernel_size, self.stride, self.pudding) # 1024 -> 256
        self.deconv5 = nn.ConvTranspose2d(2*self.ch[4], self.ch[3], self.kernel_size, self.stride, self.pudding) # 512 -> 128
        self.deconv4 = nn.ConvTranspose2d(2*self.ch[3], self.ch[2], self.kernel_size, self.stride, self.pudding) # 256 -> 64
        self.deconv3 = nn.ConvTranspose2d(2*self.ch[2], self.ch[1], self.kernel_size, self.stride, self.pudding) # 128 -> 32
        self.deconv2 = nn.ConvTranspose2d(2*self.ch[1], self.ch[0], self.kernel_size, self.stride, self.pudding) # 64 -> 16
        self.deconv1 = nn.ConvTranspose2d(2*self.ch[0], self.finalCh, self.kernel_size, self.stride, self.pudding) # 32 -> final

        self.debn1=nn.BatchNorm2d(self.finalCh)
        self.debn2=nn.BatchNorm2d(self.ch[0])
        self.debn3=nn.BatchNorm2d(self.ch[1])
        self.debn4=nn.BatchNorm2d(self.ch[2])
        self.debn5=nn.BatchNorm2d(self.ch[3])
        self.debn6=nn.BatchNorm2d(self.ch[4])

    def forward(self,x):
        
        sh = x.shape
        x = x[:,:512,:] #20 ,512 ,128 ちょっとちっちゃくする
        x = torch.unsqueeze(x,1) #20, 1, 512, 128 #チャンネル次元の追加
        
        
        x1 = self.lrelu(self.bn1(self.conv1(x)))
        x2 = self.lrelu(self.bn2(self.conv2(x1)))
        x3 = self.lrelu(self.bn3(self.conv3(x2)))
        x4 = self.lrelu(self.bn4(self.conv4(x3)))
        x5 = self.lrelu(self.bn5(self.conv5(x4)))
        x6 = self.lrelu(self.bn6(self.conv6(x5)))

        # LSTMとか追加するならここ
        z = x6

        y6 = self.lrelu(self.debn6(self.deconv6(torch.cat([z,x6],dim=1))))
        y5 = self.lrelu(self.debn5(self.deconv5(torch.cat([y6,x5],dim=1))))
        y4 = self.lrelu(self.debn4(self.deconv4(torch.cat([y5,x4],dim=1))))
        y3 = self.lrelu(self.debn3(self.deconv3(torch.cat([y4,x3],dim=1))))
        y2 = self.lrelu(self.debn2(self.deconv2(torch.cat([y3,x2],dim=1))))
        y1 = self.relu(self.debn1(self.deconv1(torch.cat([y2,x1],dim=1))))

        y0 = torch.sigmoid(y1)
        
        y0 = torch.squeeze(y0) # 20, 512, 128
        y = torch.zeros(sh)
        y[:,:512,:] = y0

        return y

def main():
    x = torch.rand([20 ,513 ,128])
    model = UnetConv2()
    y = model(x)
    print("Input data:",x.shape,"\nOutput data",y.shape)
    
if __name__ == "__main__":
    main()
