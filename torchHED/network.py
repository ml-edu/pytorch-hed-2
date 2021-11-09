# Original implementation by https://github.com/sniklaus/pytorch-hed

import torch
import torch.hub
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    """VGG-based network."""

    def __init__(self):
        super(Network, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.netScoreOne = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(128, 1, 1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(512, 1, 1, stride=1, padding=0)
        self.netScoreFiv = nn.Conv2d(512, 1, 1, stride=1, padding=0)

        self.netCombine = nn.Sequential(
            nn.Conv2d(5, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

        state_dict = torch.hub.load_state_dict_from_url(
            url='https://github.com/Davidelanz/pytorch-hed/releases/download/latest/pytorch-hed-network.pt',
            progress=False,
            file_name="pytorch-hed-network.pt")
        self.load_state_dict(state_dict, strict=False)
        #torch.save(self.state_dict, 'pytorch-hed-network.pt')

    def forward(self, tensor_in):
        blue_in = (tensor_in[:, 0:1, :, :] * 255.0) - 104.00698793
        green_in = (tensor_in[:, 1:2, :, :] * 255.0) - 116.66876762
        red_in = (tensor_in[:, 2:3, :, :] * 255.0) - 122.67891434

        tensor_in = torch.cat([blue_in, green_in, red_in], 1)

        feature1 = self.layer1(tensor_in)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)

        score1 = self.netScoreOne(feature1)
        score2 = self.netScoreTwo(feature2)
        score3 = self.netScoreThr(feature3)
        score4 = self.netScoreFou(feature4)
        score5 = self.netScoreFiv(feature5)

        sc_size = (tensor_in.shape[2], tensor_in.shape[3])

        score1 = F.interpolate(
            input=score1, size=sc_size, mode='bilinear', align_corners=False)
        score2 = F.interpolate(
            input=score2, size=sc_size, mode='bilinear', align_corners=False)
        score3 = F.interpolate(
            input=score3, size=sc_size, mode='bilinear', align_corners=False)
        score4 = F.interpolate(
            input=score4, size=sc_size, mode='bilinear', align_corners=False)
        score5 = F.interpolate(
            input=score5, size=sc_size, mode='bilinear', align_corners=False)

        return self.netCombine(
            torch.cat(
                [score1, score2, score3, score4, score5], 1)
        )
