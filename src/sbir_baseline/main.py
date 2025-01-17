import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import OursScene, SketchyScene, SketchyCOCO, Sketchy
import wandb
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Our Dataset
    train_dataset = OursScene(opts, mode='train',
        transform=dataset_transforms)
    val_dataset = OursScene(opts, mode='val',
        transform=dataset_transforms)

    # # SketchyScene Dataset
    # train_dataset = SketchyScene(opts, mode='train',
    #     transform=dataset_transforms)
    # val_dataset = SketchyScene(opts, mode='val',
    #     transform=dataset_transforms)

    # # SketchyCOCO Dataset
    # train_dataset = SketchyCOCO(opts, mode='train',
    #     transform=dataset_transforms)
    # val_dataset = SketchyCOCO(opts, mode='val',
    #     transform=dataset_transforms)

    # # Sketchy Dataset
    # train_dataset = Sketchy(opts, mode='train', transform=dataset_transforms)
    # val_dataset = Sketchy(opts, mode='val', transform=dataset_transforms)
    # val_dataset.category = 'cannon'

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    path = os.path.join(opts.checkpoint_dir, "our-dataset-epoch=1034-top10=0.53.ckpt")
    model = TripletNetwork(opts).load_from_checkpoint(checkpoint_path=path)
    # model = TripletNetwork(opts)
    wandb_key = '1cdc17e811df70a17e4d9174c95f5b4e9f4a01dc'
    _ = os.system('wandb login {}'.format(wandb_key))
    os.environ['WANDB_API_KEY'] = wandb_key
    save_path = os.path.join(opts.path_aux, 'CheckPoints', 'wandb')
    logger = WandbLogger(project=opts.project, group=opts.group, name=opts.savename, dir=save_path,
                         settings=wandb.Settings(start_method='fork'))
    
    checkpoint_callback = ModelCheckpoint(monitor="top5",
                mode="max",
                dirpath="saved_model",
                save_top_k=3,
                save_last=True,
                filename="%s-{epoch:02d}-{top10:.2f}"%opts.exp_name)

    trainer = Trainer(gpus=-1, auto_select_gpus=True, # specifies all available GPUs
                # auto_scale_batch_size=True,
                # auto_lr_find=True,
                benchmark=True,
                check_val_every_n_epoch=10,
                max_epochs=100,
                # precision=64,
                min_steps=100, min_epochs=0,
                accumulate_grad_batches=opts.grad_batches,
                # profiler="advanced",
                resume_from_checkpoint=None, # "some/path/to/my_checkpoint.ckpt"
                logger=logger,
                callbacks=[checkpoint_callback])

    print ('validating the pre-trained model...')
    trainer.validate(model, val_loader)
    top1_values = []
    # for category in val_loader.dataset.all_categories:
    #     val_loader.dataset.category = category
    #     print ('Evaluating category: ', category)
    top1_values.append(trainer.validate(model, val_loader)[0]['top1'])
    print ('Top1 score: ', np.mean(top1_values))
    input ('press any key to contrinue training')

    # trainer.tune(model)

    # trainer.fit(model, train_loader, val_loader)
    # trainer.validate(model, val_loader)
    # Retrieve model
    # checkpoint_callback.best_model_path
