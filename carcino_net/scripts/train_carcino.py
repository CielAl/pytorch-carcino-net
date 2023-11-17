"""Train the Carcino-Net with binaralized SICAPV2 data for high grade vs. rest.
"""
import argparse
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
from carcino_net.reproduce import fix_seed
from carcino_net.models.lightning_module.carcino_wrapper import CarcinoLightning
from carcino_net.models.backbone.carcino_arch import CarcinoNet
from carcino_net.dataset.lit_data import SicapDataModule
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

print(os.getcwd())
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Carcino')
fold = 2
parser.add_argument('--num_classes', default=3, type=int,
                    help='number of classes')
parser.add_argument('--export_folder', default=f'~/running_output/carcino_multi{fold}',
                    help='Export location for logs and model checkpoints')

parser.add_argument('--train_sheet_dir', default=f'/tmp/ramdisk/SICAPV2Fixed/partition/Validation'
                                                 f'/Val{fold}/Train.xlsx',
                    help='Location for the training split')
parser.add_argument('--val_sheet_dir', default='/tmp/ramdisk/SICAPV2Fixed/partition/Validation'
                                               f'/Val{fold}/Test.xlsx',
                    help='Location for the val split')


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

parser.add_argument('--focal_alpha', default=[0.03, 0.71, 0.26], type=float, nargs='+',
                    help='alpha for focal loss')
parser.add_argument('--focal_gamma', default=2, type=float,
                    help='gamma for focal loss')

parser.add_argument('--num_epochs', default=300, type=int,
                    help='max epoch')
parser.add_argument('--batch_size',
                    default=32, type=int,
                    help='batch size. '
                    'Effective batch size = N-gpu * batch size'
                    'For larger effective batch sizes, e.g., 512 or more, LARs optimizer is recommended.'
                    'In the example here only Adams is used')

parser.add_argument('--lr', default=5e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--max_t', default=90, type=int,
                    help='max_t for restarting of cosine CosineAnnealing lr_scheduler')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='weight decay')
parser.add_argument('--seed', default=31415926, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log_every_n_steps', default=25, type=int,
                    help='Log every n steps')
parser.add_argument('--gpu_index', default=[0, 1], nargs='+', type=int, help='Gpu index.')
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
                                  train_transforms=train_transform,
                                  train_pair=True,
                                  val_transforms=val_transform,
                                  val_pair=True,
                                  image_dir=opt.image_dir, mask_dir=opt.mask_dir, num_workers=opt.num_workers,
                                  batch_size=opt.batch_size,
                                  image_ext=opt.image_ext,
                                  mask_ext=opt.mask_ext, seed=opt.seed)

    data_module.setup("")
    base_model = CarcinoNet.build(n_out=opt.num_classes, img_size=(opt.patch_size, opt.patch_size),
                                  pool_sizes=opt.pool_sizes,
                                  skip_type=opt.skip_type)

    lightning_model = CarcinoLightning(base_model, num_classes=opt.num_classes,
                                       focal_alpha=opt.focal_alpha,
                                       focal_gamma=opt.focal_gamma,
                                       lr=opt.lr, batch_size=opt.batch_size)

    csv_logger = CSVLogger(save_dir=export_folder, )
    checkpoint_callbacks = ModelCheckpoint(monitor='validate_loss', save_last=True, save_top_k=3)
    # early_stop = EarlyStopping(monitor='validate_loss', patience=20, mode='min', verbose=False, check_finite=True)
    trainer = L.Trainer(accelerator='gpu', devices=opt.gpu_index,
                        callbacks=[checkpoint_callbacks, ], # early_stop
                        strategy='ddp_find_unused_parameters_true',
                        num_sanity_val_steps=0, max_epochs=opt.num_epochs, enable_progress_bar=True,
                        default_root_dir=export_folder, logger=csv_logger, precision=opt.precision,
                        use_distributed_sampler=True, sync_batchnorm=len(opt.gpu_index) > 0,
                        log_every_n_steps=opt.log_every_n_steps)
    csv_logger.log_hyperparams(opt)
    trainer.fit(lightning_model, datamodule=data_module)
