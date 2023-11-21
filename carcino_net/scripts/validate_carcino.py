"""Train the Carcino-Net with binaralized SICAPV2 data for high grade vs. rest.
"""
import argparse
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
from carcino_net.reproduce import fix_seed
from carcino_net.models.lightning_module.carcino_wrapper import CarcinoLightning
from carcino_net.models.backbone.carcino_arch import CarcinoNet
from carcino_net.dataset.lit_data import SicapDataModule
import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from carcino_net.models.callbacks.visualization import OutputWriter


print(os.getcwd())
argv = sys.argv[1:]
fold=3
parser = argparse.ArgumentParser(description='Carcino Binary')
parser.add_argument('--num_classes', default=3, type=int,
                    help='number of classes')
parser.add_argument('--train_sheet_dir', default=f'/tmp/ramdisk/SICAPV2Fixed/partition/Validation'
                                                 f'/Val{fold}/Train.xlsx',
                    help='Location for the training split')
parser.add_argument('--val_sheet_dir', default='/tmp/ramdisk/SICAPV2Fixed/partition/Validation'
                                               f'/Val{fold}/Test.xlsx',
                    help='Location for the val split')

parser.add_argument('--best_model', default='/mnt/z/UTAH/running_output/'
                                            f'carcino_multi{fold}/lightning_logs/version_2/checkpoints/'
                                            'epoch=42-step=5461.ckpt',
                    help='Location of the checkpoints of the best model')
parser.add_argument('--export_folder', default='/mnt/z/UTAH/running_output/'
                                               'carcino_multi_showcase/visualization',
                    help='Viz export location')

parser.add_argument('--image_dir', default='/tmp/ramdisk/SICAPV2Fixed/images/',
                    help='Location for the downloaded example datasets')

parser.add_argument('--image_ext', default='.jpg',
                    help='extension of image files')

parser.add_argument('--mask_dir', default='/tmp/ramdisk/SICAPV2Fixed/masks_mul/',
                    help='Location for the downloaded example datasets')

parser.add_argument('--mask_ext', default='.png',
                    help='extension of mask files')


# ------------ dataloader
parser.add_argument('--num_workers', default=8, type=int,
                    help='number of cpus used for DataLoader')
parser.add_argument('--patch_size', default=512, type=int,
                    help='Output size of RandomResizedCrop')

# ------------ trainer
parser.add_argument('--pool_sizes', default=[1, 2, 3], type=int, nargs='+',
                    help='grid sizes of spatial pyramidal pooling')
parser.add_argument('--skip_type', default='sum', type=str, choices=['sum', 'cat'],
                    help='whether to use summation or concatenation to perform skip connection')

parser.add_argument('--focal_alpha', default=0.9, type=float,
                    help='alpha for focal loss')
parser.add_argument('--focal_gamma', default=2, type=float,
                    help='gamma for focal loss')

parser.add_argument('--batch_size',
                    default=32, type=int,
                    help='batch size. '
                    'Effective batch size = N-gpu * batch size'
                    'For larger effective batch sizes, e.g., 512 or more, LARs optimizer is recommended.'
                    'In the example here only Adams is used')

parser.add_argument('--lr', default=7e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--max_t', default=90, type=int,
                    help='max_t for restarting of cosine CosineAnnealing lr_scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--seed', default=31415926, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log_every_n_steps', default=25, type=int,
                    help='Log every n steps')
parser.add_argument('--gpu_index', default=[0], nargs='+', type=int, help='Gpu index.')
parser.add_argument('--precision', default="16-mixed", type=str,
                    help='Precision configuration. Determine the precision of floating point used in training and '
                         'whether mixed precision is used.'
                         ' See https://lightning.ai/docs/pytorch/stable/common/trainer.html')


opt, _ = parser.parse_known_args(argv)

seed = opt.seed
fix_seed(seed, True, cudnn_deterministic=True)


if __name__ == "__main__":
    # dbg
    # opt.num_workers = 0
    # opt.gpu_index = [0]
    # file path curation
    export_folder = opt.export_folder
    os.makedirs(export_folder, exist_ok=True)

    train_transform = A.Compose([
        # A.RandomCrop(width=opt.patch_size, height=opt.patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
        # scale image to float within [0., 1.]. No transformation to masks
        A.ToFloat(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.ToFloat(),
        ToTensorV2(),

    ])

    data_module = SicapDataModule(train_sheet_dir=opt.train_sheet_dir, val_sheet_dir=opt.val_sheet_dir,
                                  train_transforms=None,
                                  train_pair=True,
                                  val_transforms=val_transform,
                                  val_pair=True,
                                  image_dir=opt.image_dir, mask_dir=opt.mask_dir, num_workers=opt.num_workers,
                                  batch_size=opt.batch_size,
                                  image_ext=opt.image_ext,
                                  mask_ext=opt.mask_ext, seed=opt.seed)

    data_module.setup("")
    base_model = CarcinoNet.build(n_out=opt.num_classes, img_size=(opt.patch_size, opt.patch_size), pool_sizes=opt.pool_sizes,
                                  skip_type=opt.skip_type)

    lightning_model = CarcinoLightning(base_model, focal_alpha=opt.focal_alpha,
                                       focal_gamma=opt.focal_gamma, num_classes=opt.num_classes,
                                       lr=opt.lr, batch_size=opt.batch_size)

    state_dict = torch.load(opt.best_model)['state_dict']
    lightning_model.load_state_dict(state_dict)

    writer_callbacks = OutputWriter(export_dir=opt.export_folder)

    trainer = L.Trainer(accelerator='gpu', devices=opt.gpu_index,
                        callbacks=[writer_callbacks],
                        # strategy='ddp_find_unused_parameters_true',
                        num_sanity_val_steps=0, max_epochs=1, enable_progress_bar=True,
                        default_root_dir=export_folder, logger=False, precision=opt.precision,
                        use_distributed_sampler=len(opt.gpu_index) > 1, sync_batchnorm=len(opt.gpu_index) > 1,
                        log_every_n_steps=opt.log_every_n_steps)

    trainer.predict(lightning_model, datamodule=data_module)
