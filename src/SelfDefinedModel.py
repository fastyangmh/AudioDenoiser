#import
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

#class


class SelfDefinedModel(nn.Module):
    def __init__(self, in_chans, hidden_chans, chans_scale, depth) -> None:
        super().__init__()
        self.kernel_size = 8
        self.stride = 4
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(in_channels=in_chans,
                          out_channels=hidden_chans,
                          kernel_size=self.kernel_size,
                          stride=self.stride),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_chans,
                          out_channels=hidden_chans * chans_scale,
                          kernel_size=1),
                nn.GLU(1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv1d(in_channels=hidden_chans,
                          out_channels=chans_scale * hidden_chans,
                          kernel_size=1),
                nn.GLU(1),
                nn.ConvTranspose1d(in_channels=hidden_chans,
                                   out_channels=in_chans,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride)
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_chans = hidden_chans
            hidden_chans = int(chans_scale * hidden_chans)

    def valid_length(self, length):
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        return length

    def forward(self, x):
        _, _, length = x.shape
        x = F.pad(x, (0, self.valid_length(length) - length))
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        return x[..., :length]


if __name__ == '__main__':
    #parameters
    in_chans = 1
    hidden_chans = 32
    chans_scale = 2
    depth = 5
    batch_size = 32
    sample_rate = 16000
    duration = 1

    #create model
    model = SelfDefinedModel(in_chans=in_chans,
                             hidden_chans=hidden_chans,
                             chans_scale=chans_scale,
                             depth=depth)

    #create input data
    x = torch.rand(batch_size, in_chans, sample_rate * duration)

    #get model output
    y = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y.shape))
