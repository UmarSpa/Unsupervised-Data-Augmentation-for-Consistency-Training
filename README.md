
# Unsupervised Data Augmentation for Consistency Training

This repo contains a simple and clear PyTorch implementation of the main building blocks of "[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)" by Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le


## Parameters

```
--mod:          default='semisup':          Supervised (sup) or semi-supervised training (semisup)
--sup_num:      default=4000:               Number of samples in supervised training set (out of 50K)
--val_num:      default=1000:               Number of samples in validation set (out of 50K)
--rand_seed:    default=89:                 Random seed for dataset shuffle
--sup_aug:      default=['crop', 'hflip']:  Data augmentation for supervised and unsupervised samples (crop, hflip, cutout, randaug)
--unsup_aug:    default=['randaug']:        Data augmentation (Noise) for unsupervised noisy samples (crop, hflip, cutout, randaug)
--bsz_sup:      default=64:                 Batch size for supervised training
--bsz_unsup:    default=448:                Batch size for unsupervised training
--softmax_temp: default=0.4:                Softmax temperature for target distribution (unsup)
--conf_thresh:  default=0.8:                Confidence threshold for target distribution (unsup)
--unsup_loss_w: default=1.0:                Unsupervised loss weight
--max_iter:     default=500000:             Total training iterations
--vis_idx:      default=10:                 Output visualization index
--eval_idx:     default=1000:               Validation index
--out_dir:      default='./output/':        Output directory
```

## Examples runs

For semi supervised training:
```
python main.py --mod 'semisup' --sup_num 4000 --sup_aug 'crop' 'hflip' --unsup_aug 'randaug' --bsz_sup 64 --bsz_sup 448
```

For supervised training:
```
python main.py --mod 'sup' --sup_num 49000 --sup_aug 'randaug' --bsz_sup 64
```

## Notes

Some of the code for this implementation is borrowed from online sources, as detailed below:
- Wide_ResNet in model.py: https://github.com/wang3702/EnAET/blob/73fd514c74de18c4f7c091012e5cff3a79e1ddbf/Model/Wide_Resnet.py
    - VanillaNet (initially present in guideline code) also works fine. [substitute Wide_ResNet(28, 2, 0.3, 10) with VanillaNet()]
- RandAugment in randAugment.py: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
    - my own simpler implementation of myRandAugment also works fine. [substitute RandAugment with myRandAugment]
- EMA in ema.py: https://github.com/chrischute/squad/blob/master/util.py#L174-L220
