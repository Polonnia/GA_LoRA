from .imagenet import ImageNet
from .imagenet_a import ImageNetA

def build_dataset(dataset, root_path, shots, preprocess, batch_size=32):
    if dataset == 'imagenet':
        return ImageNet(root_path, shots, preprocess, batch_size)
    elif dataset == 'imagenet-a':
        return ImageNetA(root_path, shots, preprocess, batch_size)
