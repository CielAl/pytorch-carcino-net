# pytorch-carcino-net
Pytorch implementation of [Carcino-Net](https://ieeexplore.ieee.org/document/9176235).


* Use [SICAPV2 prostate data](https://doi.org/10.1016/j.cmpb.2020.105637) for segmentation of negative (non-cancerous background), low grade (Gleason grade group 3), and high grade (Gleason grade group 4 and 5).

### Usage:

```python -m carcino_net.scripts.train_carcino --help``` for detail arguments of training scripts.



```python -m carcino_net.scripts.validate_carcino --help``` for detail arguments of validation and to export showcase output masks.


To DO:
* documentation

### Disclaimer

* Multi-class focal loss are directly derived from [Adeel Hassan's implementation](https://github.com/AdeelH/pytorch-multi-class-focal-loss]). Alternatively you may use the weighted cross entropy in pytorch combining the (1 - softmax score) as the focal term.

