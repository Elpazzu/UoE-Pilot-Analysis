# AG-UResNet50: Attention-gated Residual U-Net for Subacute Ischemic Stroke Lesion Segmentation

Implementation of a U-Net using (i) ResNet50 as encoder, (ii) attention gates on skip connections, and (iii) Leaky ReLU activation on deconvolution blocks, in order to perform subacute ischemic stroke lesion segmentation on MRI data.

## Environment

```
python = 3.9
pytorch = 2.0
torchvision = 0.15
```

## Notes

Repository structure:
```
net/att-unet.py         Implementation of attention-gated U-Net
net/attention.py        Implementation of the attention gate
net/resnet.py           Implementation of ResNet50

utils/data_prep.py      Script for augmenting ISLES 2015 SISS data
utils/dataloader.py     Script for defining Dataloader class
utils/loss.py           Script for defining loss functions & performance metrics

model_training.py       Script for training the model
model_evaluation.py     Script for evaluating the trained model
```
