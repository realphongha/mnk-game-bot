import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ResidualBlock(nn.Module):
    def __init__(self, channels, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SimpleBackbone(nn.Module):
    def __init__(self, padding, num_filters=128):
        super(SimpleBackbone, self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(2, num_filters, kernel_size=3, padding=self.padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=self.padding)

    def get_output_size(self, h_w):
        h_w = calc_conv2d_output(h_w, kernel_size=3, pad=self.padding)
        h_w = calc_conv2d_output(h_w, kernel_size=3, pad=self.padding)
        return h_w

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class ResidualNet(nn.Module):
    def __init__(self, m, n, k, in_channels=2, conv_channels=16, res_blocks=1):
        super(ResidualNet, self).__init__()
        # handle near-edge cases
        # self.padding = math.ceil(k / 2)
        self.padding = 1
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        if res_blocks:
            self.res = nn.Sequential(*[
                ResidualBlock(conv_channels, self.padding) for i in range(res_blocks)
            ])
        else:
            self.res = nn.Identity()
        self.policy = nn.Sequential(
            nn.Conv2d(conv_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2 * m * n, m * n),
        )

        self.value = nn.Sequential(
            nn.Conv2d(conv_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(m * n, 1),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.res(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


MODELS = {
    "residual": ResidualNet,
}


if __name__ == "__main__":
    import numpy as np
    from board_state.board import to_board, to_bitboard

    board = [[-1, 1 , -1, 0 ],
             [0 , -1, 0 , -1],
             [-1, 1 , 0 , 0 ],
             [1 , -1, 0 , 0 ]]
    board = np.array(board).astype(np.float32)
    m = board.shape[0]
    n = board.shape[1]
    k = 3
    board = to_board(to_bitboard(board), m, n)
    inp = torch.tensor(board).float().unsqueeze(0)
    net = ResidualNet(m, n, 3)
    print(net)
    policy, value = net(inp)
    print(policy)
    print(value)

