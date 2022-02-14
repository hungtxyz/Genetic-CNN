import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.decoder import decode


class Net(nn.Module):
    def __init__(self, bits_string, num_phase):
        # kernels = []
        super().__init__()
        input_shape = 32
        self.code = decode(bits_string, num_phase)

        # for phase, structure in enumerate(code):
        #     kernel_size = kernel_sizes[phase]
        #     for conv_th in structure:

        self.conv_list = []
        for phase in range(num_phase):
            phase_conv = []
            for i in range(len(self.code[phase]) + 1):
                if phase == 0 and i == 0:
                    phase_conv.append(nn.Conv2d(3, 8, 3, padding="same"))
                else:
                    phase_conv.append(nn.Conv2d(8, 8, 3, padding="same"))
            self.conv_list.append(nn.ModuleList(phase_conv))

        self.layers = nn.ModuleList(self.conv_list)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * int((input_shape / (2 ** num_phase)) ** 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        def link(num_operator, phase_code, phase_conv):

            if num_operator == 1:
                return F.relu(phase_conv[num_operator - 1](x))
            else:
                result = 0
                for i, bit in enumerate(phase_code[num_operator - 2]):
                    if bit == 1:
                        r = link(i + 1, phase_code, phase_conv)
                        if not isinstance(r, type(None)):
                            result += r
                a = (np.array(phase_code[num_operator - 2]) == 1).sum()
                if a == 0:
                    return None
                return F.relu(phase_conv[num_operator - 1](result))

        for i, phase in enumerate(self.layers):
            out_phase = link(len(phase), self.code[i], phase)
            x = self.pool(out_phase)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.fc3(x)


if __name__ == '__main__':
    bit_string = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1])

    net = Net(bit_string, 2, 5)
    print(net.parameters)
    # print(summary(net, (3, 32, 32)))

    test_img = np.random.randint(255, size=(1, 3, 32, 32))
    x_np = torch.from_numpy(test_img).float()
    out = net(x_np)
    print(out.shape)

