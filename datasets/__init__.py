from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenetv2 import ImageNetV2

def build_dataset(dataset, root_path, shots=0, preprocess=None, batch_size=32, num_workers=8, train_only=False):
    if dataset == 'imagenet':
        return ImageNet(
            root=root_path, 
            num_shots=shots, 
            preprocess=preprocess, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_only=train_only
        )
    elif dataset == 'imagenet-a':
        return ImageNetA(
            root=root_path, 
            num_shots=shots, 
            preprocess=preprocess, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_only=train_only
        )
    elif dataset == 'imagenet-r':
        return ImageNetR(
            root=root_path, 
            num_shots=shots, 
            preprocess=preprocess, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_only=train_only
        )
    elif dataset == 'imagenet-sketch':
        return ImageNetSketch(
            root=root_path, 
            num_shots=shots, 
            preprocess=preprocess, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_only=train_only
        )
    elif dataset == 'imagenet-v2':
        return ImageNetV2(
            root=root_path, 
            num_shots=shots, 
            preprocess=preprocess, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_only=train_only
        )