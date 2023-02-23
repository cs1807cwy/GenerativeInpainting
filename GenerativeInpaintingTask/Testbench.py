import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import PIL

if __name__ == "__main__":
    img_bgr = cv2.imread('../Example/0.jpg')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = transforms.ToPILImage()(img_rgb)
    img_pil.show()
