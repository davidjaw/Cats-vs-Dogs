import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from network import Network
from dataloader import get_dataloaders
from utils import train, validate, BinaryFocalLoss


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def str2bool(v) -> bool:
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def arg_parse(config):
    parser = argparse.ArgumentParser(description="Control dataloader and network parameters via config.")
    parser.add_argument("--dataset_path", type=str, default=config["dataset_path"], help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for dataloading.")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="Number of training epochs.")
    parser.add_argument("--use_norm", type=str2bool, default=config["use_norm"], help="Whether to normalize data or not.")
    parser.add_argument("--use_drop", type=str2bool, default=config["use_drop"], help="Whether to use dropout or not.")
    parser.add_argument("--img_size", type=int, default=config["img_size"], help="Image size for the network.")
    parser.add_argument("--network_type", type=int, default=config["network_type"],
                        help="Type of network to use (0: ResNet, 1: ResNeXt, 2: ConvNeXt-v2, "
                             "3~4: ViT + ResNeXt or ConvNeXt-v2).")
    parser.add_argument("--num_classes", type=int, default=config["num_classes"], help="Number of classes for classification.")
    parser.add_argument("--load_model", type=str2bool, default=config["load_model"], help="Whether to load a pre-trained model or not.")
    parser.add_argument("--train_mae", type=str2bool, default=config["train_mae"], help="Whether to train mae task.")
    parser.add_argument("--model_path", type=str, default=config["model_path"], help="Path to the pre-trained model.")
    parser.add_argument("--train_mode", type=str2bool, default=config["train_mode"], help="Whether to train the model")
    parser.add_argument("--use_pretrained_resnet", type=str2bool, default=config["train_mode"], help="Whether to use pre-trained ResNet")
    return parser.parse_args()


if __name__ == '__main__':
    # Load the default config from YAML
    default_config = load_config("config.yaml")
    # Override the default config with command line arguments
    args = arg_parse(default_config)

    if not args.train_mode:
        assert args.load_model, "Must load a pre-trained model for testing."
        assert args.model_path, "Must provide a path to the pre-trained model."

    # Get the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset_path, batch_size=args.batch_size,
                                                            use_norm=args.use_norm, use_drop=args.use_drop,
                                                            img_size=args.img_size)
    # Define the model
    if args.use_pretrained_resnet:
        args.network_type = 5
    model = Network(args.num_classes, network_type=args.network_type, img_size=args.img_size,
                    use_pretrained_resnet=args.use_pretrained_resnet)

    if args.load_model:
        model.load_state_dict(torch.load(args.model_path + '.h5'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.train_mode:
        print('Training mode is on, training the model.')
        criterion = BinaryFocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        writer = SummaryWriter()

        for epoch in range(args.epochs + 1):
            train_loss = train(args, epoch, model, train_loader, criterion, optimizer, device, writer)
            if epoch % 5 == 0:
                val_loss, accuracy, precision, recall = validate(args, model, val_loader, criterion, device, epoch, writer)
            print(f'Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}')

        writer.close()
    else:
        print('Training mode is off, only testing the model.')

    # Test the model
    validate(args, model, test_loader, nn.BCEWithLogitsLoss(), device, 0, None, is_test=True)

