# Pytorch Reimplementation of DeepFillv2: Free-Form Image Inpainting with Gated Convolution

Respectful and with much appreciation to ICCV 2019 Oral Paper [Free-Form Image Inpainting with Gated Convolution](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.html).

Note&Warn: this implementation is produced by a greenhand newly in deep learning, bounty of errors may occur.

It is an implementation based on PyTorch and PyTorch Lightning.

Already tested on:
```
python 3.9.16
pytorch 1.13.1
pytorch-lightning 1.9.1
GPU Nvidia GeForce GTX 1080ti
NVCC V11.7.99
```

Entry of the program is in:
```
<ProjectRoot>/Code/Main.py
```

You may use this for a quick start:
```
python Main.py fit --config "Config/config_train_example.yaml"
```

Logs are in both forms:
```
tensorboard
csv
```
Log files are saved in: (you may modify this via config YAML, see below)
```
<ProjectRoot>/Experiment/
```

Example config YAML for training and testing can be found in:
```
<ProjectRoot>/Config/
```

Many key configurations are detailly noted in:
```
<ProjectRoot>/Config/config_train_example.yaml

```
Also please refer to the Docs: [Pytorch Lighning 1.9.1](https://pytorch-lightning.readthedocs.io/en/1.9.1/starter/introduction.html)