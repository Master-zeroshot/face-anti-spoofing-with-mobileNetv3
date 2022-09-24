# Lightweight Face Anti Spoofing
Towards the solving anti-spoofing problem on RGB only data.
## Introduction
This repository contains a training and evaluation pipeline with different regularization methods for face anti-spoofing network. There are a few models available for training purposes, based on MobileNetv3 (MN3). Project supports natively three datasets: [CelebA Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof), [LCC FASD](https://arxiv.org/pdf/2003.05136.pdf).


### Data Preparation
For training or evaluating on the CelebA Spoof dataset you need to download the dataset (you can do it from the [official repository](https://github.com/Davidzhangyuanhan/CelebA-Spoof)) and then run the following script being located in the root folder of the project:
```bash
cd /data_preparation/
python prepare_celeba_json.py
```
Now you are ready to launch the training process!

### Configuration file
The script for training and inference uses a configuration file. This is [default one](./configs/config.py). You need to specify paths to datasets. The training pipeline supports the following methods, which you can switch on and tune hyperparameters while training:
* **dataset** - this is an indicator which dataset you will be using during training. Available options are 'celeba-spoof', 'LCC_FASD'.
* **multi_task_learning** - specify whether or not to train with multitasking loss. **It is available for the CelebA-Spoof dataset only!**
* **evaluation** - it is the flag to perform the assessment at the end of training and write metrics to a file
* **test_dataset** - this is an indicator on which dataset you want to test. Options are the same as for dataset parameter
* **img_norm_cfg** - parameters for data normalization
* **scheduler** - scheduler for dropping learning rate
* **data.sampler** - if it is true, then will be generated weights for `WeightedRandomSampler` object to uniform distribution of two classes
* **resize** - resize of the image
* **checkpoint** - the name of the checkpoint to save and the path to the experiment folder where checkpoint, tensorboard logs and eval metrics will be kept
* **loss** - there are available two possible losses: `amsoftmax` with `cos`, `arcos`, `cross_enropy` margins.
* **loss.amsoftmax.ratio**  - there is availability to use different m for different classes. The ratio is the weights on which provided `m` will be divided for a specific class. For example ratio = [1,2] means that m for the first class will equal to m, but for the second will equal to m/2
* **loss.amsoftmax.gamma** - if this constant differs from 0 then the focal loss will be switched on with the corresponding gamma
* **model** - there are parameters concerning model. `pretrained` means that you want to train with the imagenet weights and specify the path to it in the `imagenet weights` parameter. **size** param means the size of the mobilenetv3, there are 'large' and 'small' options. **embeding_dim** - the size of the embeding (vector of features after average pooling). **width_mult** - the width scaling parameter of the model. Note, that you will need the appropriate imagenet weights if you want to train your model with transfer learning. On google drive weights with 0.75, 1.0 values of this parameter are available
* **aug** - there are some advanced augmentations are available. You can specify `cutmix` or `mixup` and appropriate params for them. `alpha` and `beta` are used for choosing `lambda` from beta distribution, `aug_prob` response for the probability of applying augmentation on the image.
* **curves** - you can specify the name of the curves, then set option `--draw_graph` to `True` when evaluating with eval_protocol.py script
* **dropout** - `bernoulli` and `gaussian` dropouts are available with respective parameters
* **RSC** - representation self-challenging, applied before global average pooling. p, b - quantile and probability applying it on an image in the current batch
* **conv_cd** - this is the option to switch on central difference convolutions instead of vanilla one changing the value of theta from 0
* **test_steps** - if you set this parameter for some int number, the algorithm will execute that many iterations for one epoch and stop. This will help you to test all processes (train, val, test)

## Training
To start training create a config file based on the default one and run 'train.py':
```bash
python train.py --config <path to config>;
```
For additional parameters, you can refer to help (`--help`). For example, you can specify on which GPU you want to train your model. If for some reason you want to train on CPU, specify `--device` to `cpu`. The default device is `cuda 0`.

## Testing
To test your model set 'test_dataset' in config file to one of preferable dataset (available params: 'celeba-spoof', 'LCC_FASD'). Then run script:
```bash
python eval_protocol.py --config <path to config>;
```
The default device to do it is `cuda 0`.
