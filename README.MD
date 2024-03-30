# Attention-gated Residual U-Net for Subacute Ischemic Stroke Lesion Segmentation on Public Dataset

Implementation of a U-Net using (i) ResNet-50 backbone in its encoder path, (ii) attention gates on skip connections, and (iii) Leaky ReLU activation on its convolution blocks in the decoding path, in order to perform subacute ischemic stroke lesion segmentation on MRI data.

## Environment

```
python = 3.9
pytorch = 2.0
torchvision = 0.15
```

## Notes

File architecture:
```
net/att-unet.py         Implementation of attention-based U-Net
net/attention.py        Implementation of the attention gate
net/resnet.py           Implementation of ResNet-50 backbone

utils/data_prep.py      Script for augmenting ISLES2015 SISS data
utils/dataloader.py     Script for defining Dataloader class
utils/loss.py           Script for defining loss functions & performance metrics

model_training.py       Script for training the model
model_evaluation.py     Script for evaluating the trained model
```