from nnverify.specs.input_spec import InputSpecType

NET_HOME = "nnverify/nets/"
DEVICE = 'cpu'


class Args:
    def __init__(self, net, k, domain, count=None, eps=0.01, dataset='mnist', spec_type=InputSpecType.LINF, split=None,
                 pt_method=None, timeout=None, parallel=False, batch_size=10, initial_split=False, attack=None):
        self.net = NET_HOME + net
        self.domain = domain
        self.count = count
        self.eps = eps
        self.dataset = dataset
        self.spec_type = spec_type
        self.split = split
        self.pt_method = pt_method
        self.timeout = timeout
        self.parallel = parallel
        self.initial_split = initial_split
        self.attack = attack
        self.batch_size = batch_size
        self.k = k


log_file = "log.txt"
log_enabled = False


def write_log(log):
    """Appends string @param: str to log file"""
    if log_enabled:
        f = open(log_file, "a")
        f.write(log + '\n')
        f.close()


tool_name = "IVAN"
baseline = "Baseline"

# Networks
MNIST_FFN_01 = "mnist_0.1.onnx"  # 9508 neuron
MNIST_FFN_03 = "mnist_0.3.onnx"  # 12794 neurons
MNIST_FFN_L2 = "mnist-net_256x2.onnx"  # 512 neurons
MNIST_FFN_L4 = "mnist-net_256x4.onnx"   # 1024 neurons
MNIST_FFN_torch1 = "cpt/fc1.pt"

MNIST_DIFFAI_5_100 = "mnist/diffai/mnist_relu_5_100.onnx" 
MNIST_DIFFAI_5_100_FIRST_BLOCK = "mnist/diffai/model_fog_first_block_lr_1e-05_wd_0.0001_trainacc_0.9280_valacc_0.9190_testacc_0.9127.onnx" 
MNIST_DIFFAI_5_100_SECOND_BLOCK = "mnist/diffai/model_fog_second_block_lr_1e-05_wd_0.0001_trainacc_0.9790_valacc_0.9200_testacc_0.9286.onnx"
MNIST_DIFFAI_5_100_THIRD_BLOCK = "mnist/diffai/model_fog_third_block_lr_1e-05_wd_0.0001_trainacc_0.9810_valacc_0.9230_testacc_0.9295.onnx"
MNIST_DIFFAI_5_100_FORTH_BLOCK = "mnist/diffai/model_fog_fourth_block_lr_1e-05_wd_0.0001_trainacc_0.9890_valacc_0.9210_testacc_0.9288.onnx"
MNIST_DIFFAI_5_100_FIFTH_BLOCK = "mnist/diffai/model_fog_fifth_block_lr_1e-05_wd_0.0001_trainacc_0.9910_valacc_0.9210_testacc_0.9293.onnx"
MNIST_DIFFAI_5_100_ALL_BLOCK = "mnist/diffai/model_fog_all_lr_1e-05_wd_0.0001_trainacc_0.9520_valacc_0.9370_testacc_0.9425.onnx"

MNIST_PGD_6_500_1_FIRST_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx"
MNIST_PGD_6_500_1_SECOND_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx"
MNIST_PGD_6_500_1_THIRD_BLOCK = "mnist/pgd0.3/model_fog_first_block_lr_1e-07_wd_1e-06_trainacc_0.4120_valacc_0.3620_testacc_0.3882.onnx"
MNIST_PGD_6_500_1_FORTH_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx"
MNIST_PGD_6_500_1_FIFTH_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx"
MNIST_PGD_6_500_1_SIXTH_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx" 
MNIST_PGD_6_500_1_ALL_BLOCK = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx" 
MNIST_PGD_6_500_1 = "mnist/pgd0.1/ffnnRELU__PGDK_w_0.1_6_500.onnx" 

MNIST_PGD_6_500_3_FIRST_BLOCK = "mnist/pgd0.3/model_fog_first_block_lr_1e-05_wd_0.0001_trainacc_0.3320_valacc_0.2630_testacc_0.2660.onnx"
MNIST_PGD_6_500_3_SECOND_BLOCK = "mnist/pgd0.3/model_fog_second_block_lr_1e-05_wd_0.0001_trainacc_0.4850_valacc_0.4230_testacc_0.4426.onnx"
MNIST_PGD_6_500_3_THIRD_BLOCK = "mnist/pgd0.3/model_fog_third_block_lr_1e-05_wd_0.0001_trainacc_0.6140_valacc_0.5360_testacc_0.5595.onnx"
MNIST_PGD_6_500_3_FORTH_BLOCK = "mnist/pgd0.3/model_fog_fourth_block_lr_1e-05_wd_0.0001_trainacc_0.7090_valacc_0.6000_testacc_0.6282.onnx"
MNIST_PGD_6_500_3_FIFTH_BLOCK = "mnist/pgd0.3/model_fog_fifth_block_lr_1e-05_wd_0.0001_trainacc_0.8040_valacc_0.6770_testacc_0.6804.onnx"
MNIST_PGD_6_500_3_SIXTH_BLOCK = "mnist/pgd0.3/model_fog_sixth_block_lr_1e-05_wd_0.0001_trainacc_0.8240_valacc_0.6930_testacc_0.6879.onnx" 
MNIST_PGD_6_500_3_ALL_BLOCK = "mnist/pgd0.3/model_fog_all_lr_1e-05_wd_0.0001_trainacc_0.8600_valacc_0.7080_testacc_0.7151.onnx" 
MNIST_PGD_6_500_3 = "mnist/pgd0.3/ffnnRELU__PGDK_w_0.3_6_500.onnx" 

# CIFAR Networks
CIFAR_CONV_2_255 = "cifar10_2_255.onnx"  # 49402 neurons
CIFAR_CONV_8_255 = "cifar10_8_255.onnx"  # 16634 neurons
CIFAR_CONV_SMALL = "convSmall_pgd_cifar.onnx"   # 3,604 neurons
CIFAR_CONV_BIG = "convBig_diffai_cifar.onnx"  # 62,464 neurons
# CIFAR 6_200
CIFAR_6_500_BASE = "pretrained_cifar10_fcnn/ffnnRELU__PGDK_w_0.0078_6_500.onnx"
CIFAR_6_500_ALL_LAYER = "pretrained_cifar10_fcnn/model_brightness_all_lr_1e-09_wd_1e-08_trainacc_0.9820_valacc_0.5100_testacc_0.4744.onnx"
CIFAR_6_500_FIRST_LAYER = "pretrained_cifar10_fcnn/model_brightness_first_block_lr_1e-09_wd_1e-08_trainacc_0.9500_valacc_0.4800_testacc_0.4867.onnx"
CIFAR_6_500_SECOND_LAYER = "pretrained_cifar10_fcnn/model_brightness_second_block_lr_1e-05_wd_0.0001_trainacc_0.9790_valacc_0.4400_testacc_0.4933.onnx"
CIFAR_6_500_THIRD_LAYER = "pretrained_cifar10_fcnn/model_brightness_third_block_lr_1e-05_wd_0.0001_trainacc_0.9900_valacc_0.4700_testacc_0.4844.onnx"
CIFAR_6_500_FORTH_LAYER = "pretrained_cifar10_fcnn/model_brightness_fourth_block_lr_1e-08_wd_1e-07_trainacc_0.9910_valacc_0.4400_testacc_0.4856.onnx"
CIFAR_6_500_FIFTH_LAYER = "pretrained_cifar10_fcnn/model_brightness_fifth_block_lr_1e-06_wd_1e-05_trainacc_0.9950_valacc_0.4800_testacc_0.4811.onnx"
CIFAR_6_500_SIXTH_LAYER = "pretrained_cifar10_fcnn/model_brightness_sixth_block_lr_1e-09_wd_1e-08_trainacc_0.9950_valacc_0.4800_testacc_0.4778.onnx"

# OVAL21 CIFAR
CIFAR_OVAL_BASE = "oval21/cifar_base_kw.onnx"   # 3172 neurons
CIFAR_OVAL_WIDE = "oval21/cifar_wide_kw.onnx"   # 6244 neurons
CIFAR_OVAL_DEEP = "oval21/cifar_deep_kw.onnx"   # 6756 neurons
CIFAR_OVAL_BASE_T = "oval21/cifar_base_kw.pth"
CIFAR_OVAL_WIDE_T = "oval21/cifar_wide_kw.pth"
CIFAR_OVAL_DEEP_T = "oval21/cifar_deep_kw.pth"

# Randomized Smoothing models
CIFAR_RESNET_20 = "smoothing_models/cifar10/resnet20/noise_sigma/checkpoint.pth.tar"
CIFAR_RESNET_110 = "smoothing_models/cifar10/resnet110/noise_sigma/checkpoint.pth.tar"
RESNET50 = "smoothing_models/imagenet/resnet50/noise_sigma/checkpoint.pth.tar"

# Model Repair Experimentation
MNIST_3_100_UNREPAIRED = "aprnn/unrepaired/mnist_relu_3_100.pth"
MNIST_3_100_REPAIRED = "aprnn/repaired/mnist_relu_3_100.pth"
MNIST_9_100_UNREPAIRED = "aprnn/unrepaired/mnist_relu_9_100.pth"
MNIST_9_100_REPAIRED = "aprnn/repaired/mnist_relu_9_100.pth"
MNIST_9_100_REPAIRED_PT = "aprnn/repaired/mnist_relu_9_100_polytope.pth"
MNIST_9_100_REPAIRED_PTK2 = "aprnn/repaired/mnist_relu_9_100_polytopek2.pth"
MNIST_9_100_REPAIRED_PTK8 = "aprnn/repaired/mnist_relu_9_100_polytopek8.pth"
MNIST_9_100_REPAIRED_PTK16 = "aprnn/repaired/mnist_relu_9_100_polytopek16.pth"

def ACASXU(i, j):
    net_name = "acasxu/nets/ACASXU_run2a_" + str(i) + "_" + str(j) + "_batch_2000.onnx"
    return net_name
