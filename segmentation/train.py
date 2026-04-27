import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

from utils.models import FilamentSeg
from utils.datasets import HaDataset
from utils.metrics import SegmentationMetrics
from utils.criteria import FocalLoss


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write(filepath, data):
    with open(filepath, 'w') as f:
        for item in np.array(data):
            formatted_item = "{:.8f}".format(item)
            f.write(formatted_item + '\n')


def save_model(model, optimizer, epoch, save_directory, checkpoint_name):
    checkpoint = {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
    }
    save_path = save_directory + checkpoint_name
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print("model saved at " + save_path)


def test(model, checkpoint_path, test_loader, metrics, device):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
    model.eval()

    test_metrics = [0.0 for _ in range(3)]  # iou, acc, f1-score

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            pixel_metrics = metrics.pixel_metrics(outputs, labels)
            test_metrics[0] += pixel_metrics['iou']['mean']
            test_metrics[1] += pixel_metrics['accuracy']
            test_metrics[2] += pixel_metrics['f1-score']['mean']

    num = len(test_loader)
    test_metrics[0] = test_metrics[0] / num
    test_metrics[1] = test_metrics[1] / num
    test_metrics[2] = test_metrics[2] / num

    print("Test Result:\n\t"
          "Iou: {:.8f}, Accuracy: {:.8f}, F1Score: {:.8f}".format(test_metrics[0], test_metrics[1], test_metrics[2]))


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = 1
    base_channels = 40
    mid_channels = 40
    branch_channels = 20
    num_pooling = 5
    dropout_p = 0.1
    out_channels = 1
    n_heads = 2
    focal_loss_gamma = 2
    focal_loss_alpha = 0.66
    learning_rate = 2e-4
    image_size = (1024, 1024)
    batch_size = 3
    epochs = 1000
    break_epochs = 200
    output_directory = "segmentation/output/"
    dataset_name = "MHAS"
    save_directory = output_directory + "model/{}/".format(dataset_name)
    checkpoint_name = "dendritic_unet.pth"

    model = FilamentSeg(in_channels, mid_channels, out_channels, base_channels, branch_channels, n_heads=n_heads,
                        num_pooling=num_pooling, dropout_p=dropout_p, image_size=image_size).to(device)

    criterion = FocalLoss(gamma=focal_loss_gamma, alpha=focal_loss_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    metrics = SegmentationMetrics(2, 0)

    dataset_dir = "segmentation/data/{}/".format(dataset_name)
    train_dir = dataset_dir + "train/"
    val_dir = dataset_dir + "val/"
    test_dir = dataset_dir + "test/"
    train_set = HaDataset(train_dir, image_size=image_size)
    val_set = HaDataset(val_dir, image_size=image_size, test_mode=True)
    test_set = HaDataset(test_dir, image_size=image_size, test_mode=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    epoch_metrics = [[] for _ in range(8)]
    max_val_metrics = 0.0
    last_saved_epoch = 0

    for epoch in range(1, epochs + 1):
        if epoch - last_saved_epoch > break_epochs:
            print("metrics not incrementing in {} epochs, break now".format(break_epochs))
            epochs = epoch - 1
            break

        start_time = datetime.now()
        model.train()
        train_metrics = [0.0 for _ in range(4)]  # loss, iou, acc, f1-score
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_metrics[0] += loss.item()
            pixel_metrics = metrics.pixel_metrics(outputs, labels)
            train_metrics[1] += pixel_metrics['iou']['mean']
            train_metrics[2] += pixel_metrics['accuracy']
            train_metrics[3] += pixel_metrics['f1-score']['mean']

        num = len(train_loader)
        train_metrics[0] = train_metrics[0] / num / batch_size
        train_metrics[1] = train_metrics[1] / num
        train_metrics[2] = train_metrics[2] / num
        train_metrics[3] = train_metrics[3] / num

        for i in range(len(train_metrics)):
            epoch_metrics[i].append(train_metrics[i])

        model.eval()
        val_metrics = [0.0 for _ in range(4)]  # loss, iou, acc, f1-score

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_metrics[0] += loss.item()

                # outputs_list = []
                # for i in range(len(outputs)):
                #     output = outputs[i]
                #     output = output.cpu().squeeze().numpy()
                #     image = images[i]
                #     image = image.cpu().squeeze().numpy() * 255
                #     new_output = crf_postprocess(image, output, n_iters=1)
                #     output = np.round(output)
                #     new_output = np.round(new_output)
                #     output = output + new_output - output * new_output  # output & new_output
                #     outputs_list.append(torch.from_numpy(output))
                # outputs = torch.stack(outputs_list)

                pixel_metrics = metrics.pixel_metrics(outputs, labels)
                val_metrics[1] += pixel_metrics['iou']['mean']
                val_metrics[2] += pixel_metrics['accuracy']
                val_metrics[3] += pixel_metrics['f1-score']['mean']

        num = len(val_loader)
        val_metrics[0] = val_metrics[0] / num / batch_size
        val_metrics[1] = val_metrics[1] / num
        val_metrics[2] = val_metrics[2] / num
        val_metrics[3] = val_metrics[3] / num

        for i in range(len(val_metrics)):
            epoch_metrics[i + 4].append(val_metrics[i])

        end_time = datetime.now()
        print("Epoch [{}]/[{}], costed: {},\n\t"
              "TrainLoss: {:.8f}, TrainIoU: {:.8f},\n\t"
              "TrainAccuracy: {:.8f}, TrainF1Score: {:.8f}\n\t"
              "ValidationLoss: {:.8f}, ValidationIoU: {:.8f}\n\t"
              "ValidationAccuracy: {:.8f}, ValidationF1Score: {:.8f}".format(epoch, epochs,
                                                                             str(end_time - start_time),
                                                                             train_metrics[0], train_metrics[1],
                                                                             train_metrics[2], train_metrics[3],
                                                                             val_metrics[0], val_metrics[1],
                                                                             val_metrics[2], val_metrics[3]))

        new_metrics = val_metrics[1]
        if max_val_metrics <= new_metrics:
            max_val_metrics = new_metrics
            save_model(model.state_dict(), optimizer.state_dict(), epoch + 1, save_directory, checkpoint_name)
            last_saved_epoch = epoch
            test(model, None, test_loader, metrics, device)
            print("")

    test(model, save_directory + checkpoint_name, test_loader, metrics, device)
    print(checkpoint_name)

    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    plt.plot(range(1, epochs + 1), epoch_metrics[0], 'b-', label='Train Loss')
    plt.plot(range(1, epochs + 1), epoch_metrics[4], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(222)
    plt.plot(range(1, epochs + 1), epoch_metrics[1], 'b-', label='Train IoU')
    plt.plot(range(1, epochs + 1), epoch_metrics[5], 'r-', label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()

    plt.subplot(223)
    plt.plot(range(1, epochs + 1), epoch_metrics[2], 'b-', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), epoch_metrics[6], 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(224)
    plt.plot(range(1, epochs + 1), epoch_metrics[3], 'b-', label='Train F1-Score')
    plt.plot(range(1, epochs + 1), epoch_metrics[7], 'r-', label='Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training and Validation F1-Score')
    plt.legend()

    plt.savefig(output_directory + 'result.png')

    write(output_directory + 'train_loss.txt', epoch_metrics[0])
    write(output_directory + 'train_iou.txt', epoch_metrics[1])
    write(output_directory + 'train_acc.txt', epoch_metrics[2])
    write(output_directory + 'train_f1.txt', epoch_metrics[3])
    write(output_directory + 'val_loss.txt', epoch_metrics[4])
    write(output_directory + 'val_iou.txt', epoch_metrics[5])
    write(output_directory + 'val_acc.txt', epoch_metrics[6])
    write(output_directory + 'val_f1.txt', epoch_metrics[7])


if __name__ == "__main__":
    set_seed(1024)
    train()
