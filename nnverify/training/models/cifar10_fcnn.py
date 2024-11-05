import torch
import torch.nn as nn

class FullyConnected6x500(nn.Module):
    def __init__(self):
        super(FullyConnected6x500, self).__init__()
        self.Gemm_20 = nn.Linear(3072, 500)
        self.Gemm_22 = nn.Linear(500, 500)
        self.Gemm_24 = nn.Linear(500, 500)
        self.Gemm_26 = nn.Linear(500, 500)
        self.Gemm_28 = nn.Linear(500, 500)
        self.Gemm_30 = nn.Linear(500, 500)
        self.Gemm_output = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.relu(self.Gemm_20(x))
        x = self.relu(self.Gemm_22(x))
        x = self.relu(self.Gemm_24(x))
        x = self.relu(self.Gemm_26(x))
        x = self.relu(self.Gemm_28(x))
        x = self.relu(self.Gemm_30(x))
        x = self.Gemm_output(x)
        return x

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'Gemm' in k:
                parts = k.split('.')
                new_key = f"{parts[0]}.{parts[1]}"
                new_state_dict[new_key] = v
            elif 'weight' in k or 'bias' in k:
                layer_num = int(k.split('.')[0].split('_')[1])
                new_key = f"_initializer_{layer_num}_{parts[1]}"
                new_state_dict[new_key] = v
        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_initializer_'):
                parts = k.split('_')
                layer_num = int(parts[2])
                param_type = parts[3]
                if layer_num <= 12:
                    new_key = f"Gemm_{20 + (layer_num - 2) * 2}.{param_type}"
                else:
                    new_key = f"Gemm_output.{param_type}"
                new_state_dict[new_key] = v
            elif k.startswith('Gemm_'):
                new_state_dict[k] = v
        return super().load_state_dict(new_state_dict, strict)

# Function to create the model
def cifar10_6x500(in_ch=3, in_dim=32):
    return FullyConnected6x500(in_ch=in_ch, in_dim=in_dim)
