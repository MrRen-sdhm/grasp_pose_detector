class DefaultConfigs(object):
    def __init__(self, model_name):
        self.model_name = model_name

    # 1.string parameters
    train_data = "/home/sdhm/Projects/gpd2/models/gpd_dataset/12channels/3obj/train.h5"
    test_data = "/home/sdhm/Projects/gpd2/models/gpd_dataset/12channels/3obj/test.h5"
    val_data = "/home/sdhm/Projects/gpd2/models/gpd_dataset/12channels/3obj/test.h5"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"
    fold = "3obj_3ch"

    # 2.numeric parameters
    epochs = 30
    batch_size = 128
    img_channels = 12
    num_classes = 2
    lr = 0.001
    weight_decay = 0.0005
    seed = 888


# [lenet, mobilenet, mobilenet_v2, resnet, densenet, inception_v3]
config = DefaultConfigs("lenet")
