import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = channels_in, 
                      out_channels = channels_out, 
                      stride = 1, 
                      padding = 2, 
                      kernel_size = 5),
            nn.ReLU(),
            nn.InstanceNorm2d(num_features = channels_out))
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, padding = 0, stride = 2)
    def forward(self, inputs):
        out = self.layer(inputs)
        out = self.dropout(out)
        out = self.maxpool(out)
        return out
    
    
class DecoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels = channels_in,
                                            out_channels = channels_in,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            output_padding = 1)
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels = channels_in, 
                          out_channels = channels_out, 
                          stride = 1, 
                          padding = 1, 
                          kernel_size = 3),
            nn.InstanceNorm2d(num_features = channels_out))
        self.dropout = nn.Dropout(0.25)
    def forward(self, inputs):
        out = self.upsample(inputs)
        out = self.layer(out)
        out = self.dropout(out)
        return out

class ConvAE(nn.Module):
    def __init__(self, num_enc_blocks = 4, input_channels = 21):
        super(ConvAE, self).__init__()
        self.encoder_block = nn.ModuleList()
        self.decoder_block = nn.ModuleList()
        channels_in_encoder = []
        channels_out_encoder = []
        channels_in_decoder = []
        channels_out_decoder = []
        for i in range(num_enc_blocks):
            channels_in_encoder.append(2 ** i * input_channels)
            channels_out_encoder.append(2 ** (i + 1) *  input_channels)
        channels_in_decoder = channels_out_encoder[::-1]
        channels_out_decoder = channels_in_encoder[::-1]
        for i in range(num_enc_blocks):
            self.encoder_block.append(EncoderBlock(channels_in_encoder[i],
                                                   channels_out_encoder[i]))
            self.decoder_block.append(DecoderBlock(channels_in_decoder[i],
                                                   channels_out_decoder[i]))
        self.max_pool_spatial = nn.MaxPool2d(kernel_size = 2, padding = 0, stride = 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        x = inputs
        for block in self.encoder_block:
            x = block(x)
        # vector in z-space
        lattent_space = x.clone().detach()
        batch_size = lattent_space.shape[0]
        lattent_space = self.max_pool_spatial(lattent_space)
        lattent_space_mean = torch.mean(lattent_space, dim = 1)
        lattent_space_max = torch.max(lattent_space, dim = 1)[0]
        lattent_space_mean = torch.flatten(lattent_space_mean)
        lattent_space_mean = lattent_space_mean.reshape(batch_size, -1)
        lattent_space_max = torch.flatten(lattent_space_max)
        lattent_space_max = lattent_space_max.reshape(batch_size, -1)
        vectorized_lattent_space = torch.cat([lattent_space_mean, lattent_space_max], dim = 1)
        
        for block in self.decoder_block:
            x = block(x)
        x = self.sigmoid(x)
        return x, vectorized_lattent_space