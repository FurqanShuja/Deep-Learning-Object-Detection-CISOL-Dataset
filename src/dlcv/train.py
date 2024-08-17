import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch_warmup_lr import WarmupLR

from dlcv.config import get_cfg_defaults, CN
from dlcv.model import get_model
from dlcv.utils import write_results_to_csv, save_model, custom_collate_fn, get_transforms
from dlcv.training import train_and_evaluate_model
from dlcv.dataset1 import Dataset1

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.TRAIN.NO_CUDA else 'cpu')

    transform_train = get_transforms(train=True, 
                                     image_height=cfg.TRAIN.IMAGE_HEIGHT, 
                                     image_width=cfg.TRAIN.IMAGE_WIDTH, 
                                     horizontal_flip_prob=cfg.TRAIN.HORIZONTAL_FLIP_PROB, 
                                     rotation_degrees=cfg.TRAIN.ROTATION_DEGREES)
                                     
    train_dataset = Dataset1(root=cfg.TRAIN.DATA_ROOT, split='train', transform=transform_train)

    print(f"Train dataset length: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    model = get_model(num_classes=cfg.MODEL.NUM_CLASSES, model_type=cfg.MODEL.TYPE, backbone_name=cfg.MODEL.BACKBONE).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.TRAIN.BASE_LR, steps_per_epoch=len(train_loader), pct_start=cfg.TRAIN.PCT_START, anneal_strategy=cfg.TRAIN.ANNEAL_STRATEGY, epochs=cfg.TRAIN.EPOCHS)
    scheduler = WarmupLR(scheduler, cfg.TRAIN.WARMUP_LR, cfg.TRAIN.WARMUP_EPOCHS, warmup_strategy='cos')

    train_losses = train_and_evaluate_model(model, train_loader, optimizer, cfg.TRAIN.EPOCHS, device, scheduler)

    write_results_to_csv(cfg.TRAIN.RESULTS_CSV + "/" + cfg.TRAIN.RUN_NAME, train_losses)

    if cfg.TRAIN.SAVE_MODEL_PATH:
        save_model(model, cfg.TRAIN.SAVE_MODEL_PATH + "/" + cfg.TRAIN.RUN_NAME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Faster R-CNN model.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    parser.add_argument('--opts', nargs='*', help='Modify config options using the command line')

    args = parser.parse_args()

    if args.config:
        cfg = CN.load_cfg(open(args.config, 'r'))
    else:
        cfg = get_cfg_defaults()

    if args.opts:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            key_list = k.split('.')
            d = cfg
            for key in key_list[:-1]:
                d = d[key]
            d[key_list[-1]] = eval(v)

    main(cfg)
