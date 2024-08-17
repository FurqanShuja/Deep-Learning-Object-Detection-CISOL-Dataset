import argparse
import json
import os
from dlcv.config import get_cfg_defaults, CN

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def convert_to_wbf_format(results, score_threshold, weight):
    boxes = []
    scores = []
    labels = []

    for result in results:
        if result['score'] >= score_threshold:
            x1, y1, w, h = result['bbox']
            x2, y2 = x1 + w, y1 + h
            if w > 0 and h > 0:  # Ensure width and height are positive
                boxes.append([x1, y1, x2, y2])
                scores.append(result['score'] * weight)  # Adjust score by weight
                labels.append(result.get('category_id', 0))  # Default to 0 if no category_id

    return boxes, scores, labels

def custom_weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0):
    final_boxes = []
    final_scores = []
    final_labels = []

    all_boxes = [box for boxes in boxes_list for box in boxes]
    all_scores = [score for scores in scores_list for score in scores]
    all_labels = [label for labels in labels_list for label in labels]

    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)

    while sorted_indices:
        current = sorted_indices.pop(0)
        current_box = all_boxes[current]
        current_score = all_scores[current]
        current_label = all_labels[current]

        if current_score < skip_box_thr:
            continue

        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]

        for i in sorted_indices[:]:
            if all_labels[i] == current_label:
                iou = compute_iou(current_box, all_boxes[i])
                if iou >= iou_thr:
                    boxes_to_merge.append(all_boxes[i])
                    scores_to_merge.append(all_scores[i])
                    sorted_indices.remove(i)

        merged_box = [0, 0, 0, 0]
        total_score = sum(scores_to_merge)
        for box, score in zip(boxes_to_merge, scores_to_merge):
            for j in range(4):
                merged_box[j] += box[j] * score / total_score

        final_boxes.append(merged_box)
        final_scores.append(max(scores_to_merge))
        final_labels.append(current_label)

    return final_boxes, final_scores, final_labels

def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area else 0

def combine_results(filenames, weights, score_threshold, iou_threshold, use_wbf=True, predictions_path='.'):
    model_results = []
    for filename, weight in zip(filenames, weights):
        filepath = os.path.join(predictions_path, filename)
        if os.path.exists(filepath):
            model_results.append((load_json(filepath), weight))
        else:
            print(f"File {filename} does not exist. Skipping...")

    file_grouped_results = {}
    for results, weight in model_results:
        for result in results:
            file_name = result['file_name']
            if file_name not in file_grouped_results:
                file_grouped_results[file_name] = []
            file_grouped_results[file_name].append((result, weight))

    final_results = []
    for file_name, grouped_results in file_grouped_results.items():
        all_boxes = []
        all_scores = []
        all_labels = []

        for results, weight in grouped_results:
            boxes, scores, labels = convert_to_wbf_format([results], score_threshold, weight)
            if boxes:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        if not all_boxes:
            continue

        if use_wbf:
            boxes, scores, labels = custom_weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                iou_thr=iou_threshold, skip_box_thr=score_threshold
            )
        else:
            boxes = [box for sublist in all_boxes for box in sublist]
            scores = [score for sublist in all_scores for score in sublist]
            labels = [label for sublist in all_labels for label in sublist]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                final_results.append({
                    'file_name': file_name,
                    'bbox': [x1, y1, w, h],
                    'score': score,
                    'category_id': label
                })

    return final_results

def main(cfg):
    filenames = cfg.ENSEMBLE.FILENAMES
    weights = cfg.ENSEMBLE.WEIGHTS
    score_threshold = cfg.ENSEMBLE.SCORE_THRESHOLD
    iou_threshold = cfg.ENSEMBLE.IOU_THRESHOLD
    use_wbf = cfg.ENSEMBLE.USE_WBF
    predictions_path = cfg.TRAIN.PREDICTIONS

    combined_results = combine_results(filenames, weights, score_threshold, iou_threshold, use_wbf, predictions_path)
    save_json(combined_results, os.path.join(predictions_path, 'weightedfusion.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ensemble model fusion.')
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
