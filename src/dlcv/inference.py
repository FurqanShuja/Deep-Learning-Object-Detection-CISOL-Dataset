import argparse
import torch
import os
import json
from tqdm import tqdm
from torchvision.ops import nms

from dlcv.config import get_cfg_defaults, CN
from dlcv.utils import custom_collate_fn, get_transforms, json_serializable
from dlcv.dataset2 import Dataset2
from dlcv.model import get_model
from torch.utils.data import DataLoader

def evaluate_one_epoch(model, data_loader, device, save_dir, score_threshold, iou_threshold):
    model.eval()
    results = []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, filenames = batch
            images = [image.to(device) for image in images]

            outputs = model(images)
            if not outputs:
                print("No outputs from model for these images.")
                continue

            for i, output in enumerate(outputs):
                file_name = filenames[i]
                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']

                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep].cpu().numpy()
                scores = scores[keep].cpu().numpy()
                labels = labels[keep].cpu().numpy()

                score_filter = scores >= score_threshold
                boxes = boxes[score_filter]
                scores = scores[score_filter]
                labels = labels[score_filter]

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width, height = x_max - x_min, y_max - y_min
                    result = {
                        'file_name': file_name,
                        'category_id': int(label),
                        'bbox': [float(x_min), float(y_min), float(width), float(height)],
                        'score': float(score)
                    }
                    results.append(result)

    results = json_serializable(results)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_json_path = os.path.join(save_dir, "predictions.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_json_path}")

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.TRAIN.NO_CUDA else 'cpu')

    transform_test = get_transforms(train=False, 
                                    image_height=cfg.TEST.IMAGE_HEIGHT, 
                                    image_width=cfg.TEST.IMAGE_WIDTH)

    test_dataset = Dataset2(root=cfg.TRAIN.DATA_ROOT, split='test', transform=transform_test)

    print(f"Test dataset length: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    model_path = os.path.join(cfg.TRAIN.SAVE_MODEL_PATH, cfg.TRAIN.RUN_NAME + ".pth")
        # Select model based on configuration
    model = get_model(num_classes=cfg.MODEL.NUM_CLASSES, model_type=cfg.MODEL.TYPE, backbone_name=cfg.MODEL.BACKBONE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    evaluate_one_epoch(model, test_loader, device, cfg.TRAIN.PREDICTIONS, cfg.TEST.SCORE_THRESHOLD, cfg.TEST.IOU_THRESHOLD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Faster R-CNN model.')
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
