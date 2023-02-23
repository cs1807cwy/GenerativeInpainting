import os
import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union


def tensor_img_to_npimg(tensor_img):
    """
    Turn a tensor image with shape CxHxW to a numpy array image with shape HxWxC
    :param tensor_img:
    :return: a numpy array image with shape HxWxC
    """
    if not (torch.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Not supported tensor image. Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


def same_padding(images: torch.Tensor,
                 kernel_size: tuple[int, int],
                 strides: tuple[int, int],
                 dilation: tuple[int, int]):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (kernel_size[0] - 1) * dilation[0] + 1
    effective_k_col = (kernel_size[1] - 1) * dilation[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images: torch.Tensor,
                          kernel_size: tuple[int, int],
                          strides: tuple[int, int],
                          dilation: tuple[int, int]) -> torch.Tensor:
    """
    Extract patches from images and put them in the C output dimension.
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param kernel_size: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param dilation: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    batch_size, channel, height, width = images.size()
    images = same_padding(images, kernel_size, strides, dilation)
    unfold = torch.nn.Unfold(kernel_size=kernel_size,
                             dilation=dilation,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(image_height_width: tuple[int, int],
                mask_height_width: tuple[int, int],
                margin_vertical_horizontal: tuple[int, int]
                ) -> tuple[int, int, int, int]:
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    image_height, image_width = image_height_width
    h, w = mask_height_width
    margin_height, margin_width = margin_vertical_horizontal
    maxt = image_height - margin_height - h
    maxl = image_width - margin_width - w
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)
    return (t, l, h, w)


def bbox2mask(bbox: tuple[int, int, int, int],
              image_height_width: tuple[int, int],
              max_delta_height_width: tuple[int, int]) -> torch.Tensor:
    """Generate mask tensor from bbox.

    Args:
        bbox: tuple, (top, left, height, width)
        max_delta_height_width: (max_delta_h, max_delta_w) used for randomly shrinking mask bbox

    Returns:
        torch.Tensor: output with shape [B=1, C=1, H, W]

    """
    image_height, image_width = image_height_width
    max_delta_h, max_delta_w = max_delta_height_width
    mask = torch.zeros((1, 1, image_height, image_width)).float()
    delta_h = np.random.randint(max_delta_h // 2 + 1)
    delta_w = np.random.randint(max_delta_w // 2 + 1)
    # mask: 1 for invalid, 0 for valid
    mask[:, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask


def brush_stroke_mask(image_height_width: tuple[int, int]) -> torch.Tensor:
    """Generate mask tensor from bbox.

    Returns:
        torch.Tensor: output with shape [B=1, C=1, H, W]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * np.pi / 5
    angle_range = 2 * np.pi / 15
    min_width = 12
    max_width = 40

    def generate_mask(H, W):
        average_radius = np.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * np.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, 1, H, W))
        return mask

    mask = torch.from_numpy(generate_mask(*image_height_width))
    return mask


def mask_image(input: torch.Tensor,
               image_height_width: Optional[tuple[int, int]],
               mask_height_width: tuple[int, int],
               margin_vertical_horizontal: tuple[int, int],
               max_delta_height_width: tuple[int, int]
               ):
    if image_height_width is None:
        image_height_width = (input.shape[2], input.shape[3])
    else:
        input = F.interpolate(input, image_height_width)
    bbox = random_bbox(image_height_width, mask_height_width, margin_vertical_horizontal)

    # mask: 1 for invalid, 0 for valid
    regular_mask = bbox2mask(bbox, image_height_width, max_delta_height_width).type_as(input)
    irregular_mask = brush_stroke_mask(image_height_width).type_as(input)
    mask = (regular_mask.eq(1.) + irregular_mask.eq(1.)).float()
    result = input * (1. - mask)
    return result, mask


def Test_random_bbox():
    print('[Test] random_bbox:')
    image_shape = (256, 256)
    mask_shape = (128, 128)
    margin = (0, 0)
    bbox = random_bbox(image_shape, mask_shape, margin)
    print(f'bbox\t -> \ttlhw: {bbox}')


def Test_bbox2mask():
    print('[Test] bbox2mask:')
    image_shape = (256, 256)
    mask_shape = (128, 128)
    margin = (0, 0)
    max_delta_shape = (32, 32)
    bbox = random_bbox(image_shape, mask_shape, margin)
    mask = bbox2mask(bbox, image_shape, max_delta_shape)
    print(f'mask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')

    import matplotlib.pyplot as plt
    mask = mask.cpu()
    plt.figure("Bbox-Mask")
    plt.title("Bbox-Mask")
    plt.imshow(mask.squeeze(dim=0).permute(1, 2, 0), cmap='gray')
    plt.show()


def Test_brush_stroke_mask():
    print('[Test] brush_stroke_mask:')
    image_shape = (256, 256)
    mask = brush_stroke_mask(image_shape)
    print(f'mask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device} (always on cpu!)')

    import matplotlib.pyplot as plt
    mask = mask.cpu()
    plt.figure("Brush-Mask")
    plt.title("Brush-Mask")
    plt.imshow(mask.squeeze(dim=0).permute(1, 2, 0), cmap='gray')
    plt.show()


def Test_mask_image():
    print('[Test] mask_image:')
    image_shape = (512, 512)
    mask_shape = (128, 128)
    margin = (100, 100)
    max_delta_shape = (50, 50)
    image = Image.open('./Example/0.jpg')
    import torchvision.transforms as transforms
    image = transforms.ToTensor()(image).unsqueeze(0)
    masked_image, mask = mask_image(image, image_shape, mask_shape, margin, max_delta_shape)
    print(f'image\t -> \tshape: {image.shape}, dtype: {image.dtype}, device: {image.device}')
    print(f'mask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    print(
        f'masked_image\t -> \tshape: {masked_image.shape}, dtype: {masked_image.dtype}, device: {masked_image.device}')
    import matplotlib.pyplot as plt
    image = image.cpu()
    mask = mask.cpu()
    masked_image = masked_image.cpu()
    plt.figure("Example")
    plt.title("Example")
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.show()
    plt.figure("Mask")
    plt.title("Mask")
    plt.imshow(mask.squeeze(0).permute(1, 2, 0), cmap='gray')
    plt.show()
    plt.figure("Masked Image")
    plt.title("Masked Image")
    plt.imshow(masked_image.squeeze(0).permute(1, 2, 0))
    plt.show()

    # Batch test
    image = torch.rand(2, 3, 5, 7)
    masked_image, mask = mask_image(image, image_shape, mask_shape, margin, max_delta_shape)
    print(f'image\t -> \tshape: {image.shape}, dtype: {image.dtype}, device: {image.device}')
    print(f'mask\t -> \tshape: {mask.shape}, dtype: {mask.dtype}, device: {mask.device}')
    print(
        f'masked_image\t -> \tshape: {masked_image.shape}, dtype: {masked_image.dtype}, device: {masked_image.device}')


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    Test_random_bbox()
    Test_bbox2mask()
    Test_brush_stroke_mask()
    Test_mask_image()
