# Cat vs Dog Classifier
This project offers a solution to the classic Cats vs Dogs classification challenge, using the Kaggle's Cats vs Dogs dataset. The model is built upon a variety of network architectures, with an emphasis on the pre-trained ResNet.

## Dataset Details
The Kaggle dataset doesn't provide labels for the test set. Hence, we've split the test set from the original training set at a 70:30 ratio. For details on this split, see this  [code segment](https://github.com/davidjaw/Cats-vs-Dogs/blob/main/dataloader.py#L144-L151).

## Key Highlights
1. **Masked Autoencoder Integration**: Although we derive inspiration from the concept of masked autoencoder[^1], our implementation closely mirrors ConvNeXtv2[^2].
2. **Network Architecture Implementation**: The project explores various network architectures such as ResNet, ResNeXt, ConvNeXtv2, and combinations of CNN and ViT. The specific implementations can be found in `network.py`.
3. **Enhanced Loss Function**: We utilize a combination of BCE loss and Focal loss[^3] to boost model performance.


### Ablation Study: MAE and Pixel Drop
* In this experiment, we set the `epochs` to 50 and `network_type` to 1 (ResNeXt-like network) since it's the most lightweight network.
* The following presents an ablation study on the influence of MAE and pixel drop during training:
![](https://raw.githubusercontent.com/davidjaw/Cats-vs-Dogs/main/img/comparison_resnext.png)
* And the result for the test set:

| Network type | MAE   | Drop | Accuracy | Precision | Recall |
|--------------|-------|------|----------|-----------|--------|
| 1            | ❌    | ❌   | 89.39%   | 94.27%    | 83.87% |
| 1            | ✅    | ❌   | 90.73%   | 93.71%    | 87.33% |
| 1            | ❌    | ✅   | 87.12%   | 85.89%    | 88.83% |
| 1            | ✅    | ✅   | 86.89%   | 89.72%    | 83.33% |

In summary, despite theoretical expectations, pixel drop augmentation does not appear beneficial at 50 epochs in our experiments. 
While the MAE task shows potential aids during training, this improvement is not evident in the test set results.

[^1]: Masked Autoencoders Are Scalable Vision Learners, https://arxiv.org/abs/2111.06377
[^2]: ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders, https://arxiv.org/abs/2301.00808
[^3]: Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002

## Requirements
* pytorch > 2.0
* torchvision
* torchmetrics
* matplotlib
* tensorboard

## Usage
1. **Setup**: Download the Kaggle's Cats vs Dogs dataset and place it in the desired directory.
2. **Configuration**: Modify the `config.yaml` file to adjust the training parameters if necessary.
   * Note: CLI arguments will override the config file parameters.
3. **Exec**: Run the script `main.py`

## Model performance
Finetune the pretrained model always provide >95% accuracy, longer probably better:
```shell
python main.py --epochs 15 --use_pretrained_resnet 1 --dataset_path <path_to_dataset>
```

![](https://raw.githubusercontent.com/davidjaw/Cats-vs-Dogs/main/img/result.png)

To test the trained model saved as `weights/model.h5`, use:
```shell
python main.py --dataset_path <path_to_dataset> --train_mode 0 --use_pretrained_resnet 1 --model_path weights --load_model 1 --model_name model
```

### Configuration details
The default configurations can be found in `config.yaml` and this [code segment](https://github.com/davidjaw/Cats-vs-Dogs/blob/main/main.py#L26-L44).
The following parameters are also available via CLI arguments:

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

## Acknowledgements
We'd like to acknowledge [ConvNeXtv2](https://github.com/facebookresearch/ConvNeXt-V2) as some parts of our implementation are inspired or copy from their work.


