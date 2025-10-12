from .imagenet import ImageNet


def build_dataset(dataset, root_path, shots, preprocess):
    return ImageNet(root_path, shots, preprocess)