import torch
import torch.nn as nn
import torchvision

class UNet(nn.Module):
    def __init__(self, encoder='resnet34', num_classes=3, pretrained=True):
        super().__init__()
        self.encoder = self._get_encoder(encoder, pretrained)
        self.decoder = UNetDecoder(self.encoder.num_channels)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _get_encoder(self, name, pretrained):
        if name == 'resnet34':
            encoder = torchvision.models.resnet34(pretrained=pretrained)
            return ResNetEncoder(encoder)
        # Add more encoder options if needed
        
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.final(decoder_output)

class ResNetEncoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.num_channels = [64, 128, 256, 512]
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
    def forward(self, x):
        x0 = self.firstconv(x)
        x0 = self.firstbn(x0)
        x0 = self.firstrelu(x0)
        x1 = self.firstmaxpool(x0)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        return [x0, x1, x2, x3, x4]

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(encoder_channels[-1], encoder_channels[-2]),
            DecoderBlock(encoder_channels[-2], encoder_channels[-3]),
            DecoderBlock(encoder_channels[-3], encoder_channels[-4]),
            DecoderBlock(encoder_channels[-4], 64)
        ])
        
    def forward(self, features):
        features = features[::-1]  # Reverse for decoder
        x = features[0]
        skips = features[1:]
        
        for decoder_block, skip in zip(self.decoder_blocks, skips):
            x = decoder_block(x, skip)
            
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
