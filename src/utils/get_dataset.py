import my_model


def get_dataset(dataset):
    """
    dataset:
        imagenet:
            ImageNet train dataset
        imagenet-all, imagenet-test:
            ImageNet validation dataset
        imagenet-training:
            ImageNet train dataset using data augumentation
        imagenet100-training:
            ImageNet100 train dataset using data augumentation
        imagenet100-training:
            ImageNet100 train dataset using data augumentation
        cub:
            Caltech-UCSD Birds 200
    """
    if dataset == "imagenet":
        dataset = my_model.get_ilsvrc2012()
        n_classes = 1000

    elif "imagenet100" in dataset and "training" in dataset:
        dataset = my_model.get_imagenet100(mode="train", transform_type="vgg_train")
        n_classes = 100
    elif "imagenet100" in dataset and "test" in dataset:
        dataset = my_model.get_imagenet100(mode="test")
        n_classes = 100
    elif "imagenet" in dataset and "all" in dataset:
        dataset = my_model.get_ilsvrc2012(mode="test")
        n_classes = 1000
    elif "imagenet" in dataset and "test" in dataset:
        dataset = my_model.get_ilsvrc2012(mode="test")
        n_classes = 1000
    elif "imagenet" in dataset and "training" in dataset:
        dataset = my_model.get_ilsvrc2012(mode="train", transform_type="vgg_train")
        n_classes = 1000
    elif dataset == "cub":
        dataset = my_model.get_CUB()
        n_classes = 200
    else:
        raise ValueError("Unknow dataset: {}".format(dataset))

    return dataset, n_classes
