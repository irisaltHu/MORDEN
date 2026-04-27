import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from utils.models import FilamentSeg
from utils.datasets.transforms import ImageTransforms
from utils.post_processing.tools import crf_postprocess, segmentation_compare, fill_hole_and_denoise, edge_detection


def inference_dataset(model, data_loader, device, show=False):
    """
    :param model: model
    :param data_loader: requires DataLoader.
    :param device: device
    :param show: show the result or not
    :return: returns list of inference result(numpy type)
    """
    model.eval()
    ret = []
    with torch.no_grad():
        for images, _ in data_loader:
            print(images.shape)
            images = images.to(device)

            outputs = model(images)

            for output in outputs:
                output = torch.round(output)
                output = output.cpu().squeeze().numpy() * 255
                output = output.astype(np.uint8)
                ret.append(output)

                if show:
                    plt.imshow(255 - output, cmap='gray')
                    plt.show()

    return ret


def inference_images(model, image_dir, device, batch_size,
                     image_size=(1024, 1024), show=False, save_dir=None, imread_config=cv2.IMREAD_GRAYSCALE,
                     out_image_size=(1024, 1024)):
    """
    :param model: model
    :param image_dir: requires string. directory of the images
    :param device: device
    :param batch_size: batch size
    :param image_size: requires tuple. image size of segmentation model
    :param show: show or not
    :param save_dir: requires string. directory to save the predictions
    :param imread_config: config of cv2.imread
    :param out_image_size: requires tuple. image size for DenseCRF
    :return: returns list of inference result(numpy type)
    """
    image_names = os.listdir(image_dir)
    images = []
    origin_images = []
    for image_name in image_names:
        if not image_name.endswith("png") and not image_name.endswith(".jpg"):
            continue
        image_path = image_dir + image_name
        image = cv2.imread(image_path, imread_config)
        origin_images.append(image)

        trans = ImageTransforms(image)
        trans.resize(image_size)
        trans.min_max_normalization()
        trans.to_tensor()

        image = trans.data
        images.append(image)

    images = torch.stack(images)
    images = images.to(device)

    model.eval()
    ret = []
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2
    solar_r = min(center_x, center_y) - 1
    y, x = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) >= (solar_r ** 2)
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            outputs = model(images[i:i + batch_size])

            for j in range(len(outputs)):
                cnt += 1
                output = outputs[j]

                # output = torch.round(output)
                # output = output.cpu().squeeze().numpy() * 255
                output = output.cpu().squeeze().numpy()
                # output = output.astype(np.uint8)

                output[circle_mask] = 0
                output = cv2.resize(output, out_image_size)
                image = cv2.resize(origin_images[i + j], out_image_size)
                new_output = crf_postprocess(image, output, alpha=8, beta=12, gamma=3, n_iters=10)
                output = np.round(output)
                new_output = np.round(new_output)
                output = output + new_output - output * new_output
                output[output > 1.] = 1.
                output = output * 255
                output = output.astype(np.uint8)
                output = fill_hole_and_denoise(output)
                output[output > 0] = 255

                ret.append(output)

                if show:
                    plt.imshow(255 - output, cmap='gray')
                    plt.show()

                if save_dir is not None:
                    print("saving " + save_dir + image_name)
                    path = Path(save_dir)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.mkdir(exist_ok=True)
                    image_name = image_names[i + j]
                    # cv2.imwrite(save_dir + image_name, output)
                    Image.fromarray(output).convert('L').save(save_dir + image_name)

    return ret


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = 1
    base_channels = 40
    mid_channels = 40
    branch_channels = 20
    num_pooling = 5
    dropout_p = 0.1
    n_heads = 2
    out_channels = 1
    image_size = (1024, 1024)
    out_image_size = (2048, 2048)
    batch_size = 3

    model = FilamentSeg(in_channels, mid_channels, out_channels, base_channels, num_pooling=num_pooling,
                        n_heads=n_heads,
                        branch_channels=branch_channels, dropout_p=dropout_p, image_size=image_size).to(device)

    checkpoint_path = "segmentation/output/model/MHAS/dendritic_unet.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # inference_dir = "/segmentation/data/MHAS/test/"
    # dataset = HaDataset(inference_dir, image_size=image_size, test_mode=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # inference_dataset(model, data_loader, device)

    image_dir = "segmentation/data/MHAS/test/images/"
    save_dir = "output/MHAS/test/seg/"
    inference_images(model, image_dir, device, batch_size, save_dir=save_dir, image_size=image_size,
                     out_image_size=out_image_size)


if __name__ == "__main__":
    main()
