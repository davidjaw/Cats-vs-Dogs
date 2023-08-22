# Cat vs Dog Classifier

This repository contains a solution to the Cats vs Dogs Classification challenge using the Kaggle's Cats vs Dogs dataset. The best model is based on the pre-trained ResNet.

Note that since the dataset does not release the label for the test set, the test set is split from the training set with a ratio of 0.2. The test set is used to evaluate the model performance.
Dataset split can refer to this [code segment](https://github.com/davidjaw/Cats-vs-Dogs/blob/main/dataloader.py#L144-L151).

Some of the code is borrowed 

## What special things are in this repository?
1. I use the concept of masked autoencoder[^1] during training, however, the implementation is more similar to the way ConvNeXtv2[^2] did.
2. I tried several network architecture like ResNet, ResNeXt, ConvNeXtv2, or the combination of CNN and ViT. Implementation details can be found in `network.py`.

[^1]: Masked Autoencoders Are Scalable Vision Learners, https://arxiv.org/abs/2111.06377
[^2]: ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders, https://arxiv.org/abs/2301.00808

## Usage
1. Download the dataset from Kaggle's Cats vs Dogs dataset and place it in an appropriate directory.
2. Modify the `config.yaml` file to adjust the training parameters if necessary.
   * parameters in config file will be overwritten by command line arguments
3. Run the script `main.py`

## Requirements
* pytorch > 2.0
* torchvision
* torchmetrics
* matplotlib
* tensorboard

## Best model
Finetune the pretrained model always provide >94% accuracy:
```shell
python main.py --epochs 15 --use_pretrained_resnet True
```

Test on trained model at `weights/model.h5`:
```shell
python main.py --train_mode False --use_pretrained_resnet True --model_path weights --load_model True
```

### Configurations
Check `config.yaml` for the default configuration parameters. 

The following parameters are available for command line arguments:
* `dataset_path`: Path to the dataset.
* `batch_size`: Batch size for data loading.
* `epochs`: Number of training epochs.
* `use_norm`: Whether to normalize data or not.
* `use_drop`: Whether to use dropout on image or not.
* `img_size`: Image size for the network.
* `network_type`: Type of network to use (Refer to arg_parse for choices).
  * `0`: ResNet-like network.
  * `1`: ResNeXt-like network.
  * `2`: ConvNeXtv2-like network.
  * `3`: A combination of ResNeXt and ViT.
  * `4`: A combination of ConvNeXtv2 and ViT.
  * `5`: Pretrained ResNet50.
* `num_classes`: Number of classes for classification.
* `load_model`: Whether to load a pre-trained model or not.
* `train_mae`: Whether to train the mae task.
* `model_path`: Path to the pre-trained model.
* `train_mode`: Whether to train the model.
* `use_pretrained_resnet`: Whether to use a pre-trained ResNet.
  * Note that this overrides the `network_type` parameter.





