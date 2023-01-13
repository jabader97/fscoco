from options import opts
import torch
from torchvision import transforms
from utils import collate_fn
from model import Photo2Sketch
from dataloader import OursScene


def setup_logger(args):
    import os
    import wandb
    _ = os.system('wandb login {}'.format(args.wandb_key))
    os.environ['WANDB_API_KEY'] = args.wandb_key
    save_path = os.path.join(args.path_aux, 'CheckPoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    group_name = args.group
    if args.savename == 'group_plus_seed':
        args.savename = group_name
    wandb.init(project=args.project, group=group_name, name=args.savename, dir=save_path,
               settings=wandb.Settings(start_method='fork'))
    wandb.config.update(vars(args))


dataset_transforms = transforms.Compose([
    transforms.Resize((opts.max_len, opts.max_len)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = OursScene(opts, mode='train', transform=dataset_transforms)   
dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=opts.batch_size, collate_fn=collate_fn)

dataset_val = OursScene(opts, mode='val', transform=dataset_transforms)   
dataloader_val = torch.utils.data.DataLoader(
    dataset=dataset_val, batch_size=opts.batch_size, collate_fn=collate_fn)

if opts.log_online:
    setup_logger(opts)


model = Photo2Sketch()
if torch.cuda.is_available():
    model = model.cuda()
# model.load_state_dict(torch.load('model_small_sketch.ckpt'))

for epoch in range(1600, 100000):
    if epoch % 100 == 0:
        # First evaluate
        print ('evaluation started...')
        output_dir = 'output_small_sketch/%d'%epoch
        model.eval()
        for batch_idx, batch in enumerate(dataloader_val):
           model.evaluate(batch, batch_idx, output_dir)
        print ('evaluation done. Check %s'%output_dir)

    # Train model
    model.train()
    for params in model.img_encoder.parameters():
        params.require_grad = False

    for batch_idx, batch in enumerate(dataloader_train):
        loss = model.train_batch(batch)
        if batch_idx % 30 == 0:
            print ('Epoch: {}, Iter: {}/{}, Loss: {}'.format(
                epoch, batch_idx, len(dataloader_train), loss))

    if epoch % 200 == 0:
        torch.save(model.state_dict(), 'model_small_sketch.ckpt')
