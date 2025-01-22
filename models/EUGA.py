import torch.nn as nn
import torch
class EUGA(nn.Module):
    def __init__(self, in_channel=3, out_channel=None, ratio=4):
        super(EUGA, self).__init__()
        self.chanel_in = in_channel
        if out_channel is None:
            out_channel = in_channel // ratio if in_channel // ratio > 0 else 1

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Conv2d(1, in_channel, kernel_size=1)

        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.chanel_in)
        # 256x256
        self.fc = nn.Linear(16384,16384)
        # 128x128
        # self.fc = nn.Linear(4096,4096)

    def forward(self, fa, fb):
        """

        :param fa: Input
        :param fb: Uncertainity map
        :return:
        """
        B, C, H, W = fa.size()
        fb = self.pool(fb)
        fb = self.up(fb)

        proj_query = self.query_conv(fb).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C

        proj_key = self.key_conv(fb).view(B, -1, H * W)  # B, C, HW

        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW

        attention = self.softmax(energy)  # B, HW, HW
        proj_value = self.value_conv(fa).view(B, -1, H * W)  # B, C, HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, C, HW
        out = self.fc(out)
        out = out.view(B, C, H, W)

        out = out + fa

        out = self.relu(out)

        return out

if __name__ == '__main__':
    block = EUGA(in_channel=3, ratio=4)
    fa = torch.rand(16, 3, 32, 32)
    fb = torch.rand(16, 3, 32, 32)

    output = block(fa, fb)
    print(fa.size())
    print(fb.size())
    print(output.size())