"""Wrapper for the models module."""

import alexnet
import fully_connected
import inception_v3
import resnet50
import vgg
import vit

# %%
if __name__ == "__main__":
    # instantiate each model
    alexnet_model = alexnet.AlexNet()
    fully_connected_model = fully_connected.FullyConnected(
        num_layers=1, num_units=100, num_features=784, num_classes=10
    )
    inception_v3_model = inception_v3.InceptionV3()
    resnet50_model = resnet50.ResNet50()
    vgg_model = vgg.VGG()
    vit_model = vit.ViT(num_classes=10)

# %%
