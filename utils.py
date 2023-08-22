import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import itertools
from dataloader import denormalize
import matplotlib.pyplot as plt


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.normal_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        # Compute the sigmoid of the inputs
        probs = torch.sigmoid(inputs)
        normal_BCE = self.normal_loss(probs, targets)

        # Compute the focal loss components
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - pt).pow(self.gamma) * pt.log()

        loss = loss + normal_BCE
        # Handle the reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def train(args, epoch, model, train_loader, criterion, optimizer, device, writer):
    model.train()  # Set the model to training mode
    total_bce_loss = 0.0
    total_mae_loss = 0.0

    mae_criterion = nn.L1Loss()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, reconstruct = model(inputs)
        bce_loss = criterion(outputs, labels)
        if args.train_mae:
            in_img = denormalize(inputs, device)
            mae_loss = mae_criterion(reconstruct, in_img)
            total_mae_loss += mae_loss.item()
            loss = bce_loss + mae_loss
        else:
            loss = bce_loss
        loss.backward()
        optimizer.step()

        total_bce_loss += bce_loss.item()

    average_loss = total_bce_loss / len(train_loader)
    writer.add_scalar('Train/BCE_Loss', average_loss, epoch)
    if args.train_mae:
        writer.add_scalar('Train/MAE_Loss', total_mae_loss / len(train_loader), epoch)
    return average_loss


def validate(args, model, val_loader, criterion, device, epoch, writer, is_test=False):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.append(outputs)
            all_labels.append(labels)

    average_loss = total_loss / len(val_loader)

    if epoch == 0 and model.network_type < 3 and not is_test:
        # Record the graph of the model only once (ViT will cause error, don't know why)
        writer.add_graph(model, inputs)

    all_outputs = torch.cat(all_outputs, 0)
    all_labels = torch.cat(all_labels, 0)
    probs = torch.nn.functional.sigmoid(all_outputs)
    pred_classes = (probs > 0.5).long()

    # Calculate metrics
    accuracy = torchmetrics.functional.accuracy(pred_classes, all_labels, task='binary', num_classes=args.num_classes)
    precision = torchmetrics.functional.precision(pred_classes, all_labels, num_classes=args.num_classes, task='binary')
    recall = torchmetrics.functional.recall(pred_classes, all_labels, num_classes=args.num_classes, task='binary')
    conf_matrix = torchmetrics.functional.confusion_matrix(pred_classes, all_labels, num_classes=args.num_classes, task='binary')
    # For ROC curve, considering a binary classification
    fpr, tpr, _ = torchmetrics.functional.roc(all_outputs, all_labels.long(), num_classes=args.num_classes, task='binary')

    if is_test:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Accuracy: {accuracy * 100:.2f}, Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}',
                     y=1.02)
        plot_confusion_matrix(conf_matrix, classes=[i for i in range(args.num_classes)], ax=ax1)
        plot_roc_curve(fpr, tpr, ax=ax2)
        fig.tight_layout()
        fig.savefig('result.png')
        print(f'\t[Test] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    else:  # Validation phase
        print(f'\t[Valid] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        # Logging metrics to TensorBoard every 5 epochs
        writer.add_scalar('Valid/Loss', average_loss, epoch)
        writer.add_scalar('Valid/Accuracy', accuracy, epoch)
        writer.add_scalar('Valid/Precision', precision, epoch)
        writer.add_scalar('Valid/Recall', recall, epoch)
        writer.add_figure('Valid/Confusion_Matrix',
                          plot_confusion_matrix(conf_matrix, classes=[i for i in range(args.num_classes)]), epoch)
        writer.add_figure('Valid/ROC_Curve', plot_roc_curve(fpr, tpr), epoch)

        torch.save(model.state_dict(), f'{args.model_path}-{epoch}.h5')

    return average_loss, accuracy, precision, recall


def plot_confusion_matrix(cm, classes, ax=None):
    cm = cm.cpu().numpy()
    # This function plots the confusion matrix using matplotlib and returns the plot as a figure
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax


def plot_roc_curve(fpr, tpr, ax=None):
    fpr = fpr.cpu().numpy()
    tpr = tpr.cpu().numpy()
    # This function plots the ROC curve using matplotlib and returns the plot as a figure
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    return ax
