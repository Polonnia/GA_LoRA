from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenetv2 import ImageNetV2

def build_dataset(dataset, root_path, shots=0, preprocess=None, batch_size=32):
    if dataset == 'imagenet':
        return ImageNet(root_path, shots, preprocess, batch_size)
    elif dataset == 'imagenet-a':
        return ImageNetA(root_path, shots, preprocess, batch_size)
    elif dataset == 'imagenet-r':
        return ImageNetR(root_path, shots, preprocess, batch_size)
    elif dataset == 'imagenet-sketch':
        return ImageNetSketch(root_path, shots, preprocess, batch_size)
    elif dataset == 'imagenet-v2':
        return ImageNetV2(root_path, shots, preprocess, batch_size)
