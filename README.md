# Cat vs Dog Classifier

This repository contains a solution to the Cats vs Dogs Classification challenge using the Kaggle's Cats vs Dogs dataset. The best model is based on the pre-trained ResNet.

## Requirements
* pytorch > 2.0
* torchvision
* torchmetrics
* matplotlib
* tensorboard

## Usage
1. Download the dataset from Kaggle's Cats vs Dogs dataset and place it in an appropriate directory.
2. Modify the `config.yaml` file to adjust the training parameters if necessary.
3. Run the script `main.py`

### Best model
* From train to test
```shell
python main.py --epochs 15 --use_pretrained_resnet
```

### Configuration
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
* `train_mae`: Whether to train mae task.
* `model_path`: Path to the pre-trained model.
* `train_mode`: Whether to train the model.
* `use_pretrained_resnet`: Whether to use a pre-trained ResNet.
  * Note that this overrides the `network_type` parameter.





