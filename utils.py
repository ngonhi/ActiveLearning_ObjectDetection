from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib.pyplot import imshow
from tqdm import tqdm
from IPython.display import display


with open('../TFS_analyze/TFS_vinai_batch4/labels.json') as f:
    labels = json.load(f)

with open('../TFS_analyze/TFS_vinai_batch4/val_5k.json') as f:
    pred = json.load(f)

fname2id = {}
for label in labels['images']:
    fname2id[label['file_name']] = label['id']
    
id2fname = {v: k for k, v in fname2id.items()}


def create_im_dict(pred_results):
    '''
    Given prediction results, create a dictionary with key being image id.
    Each value is a dictionary of confidence score grouped into classes.

    Return:
    {'image_id1': {'class0': [score1, score2, score3, ...],
                   'class1': [score1, score2, score3, ...],
                   ...,
                   'class5': [score1, score2, score3, ...]},
     'image_id2': ...}
    '''
    ret = {}
    for pred in pred_results:
        image_id = pred['image_id']
        cat = pred['category_id']
        if image_id not in ret:
            ret[image_id] = {0: [],
                             1: [],
                             2: [],
                             3: [],
                             4: [],
                             5: []}
        ret[image_id][cat].append(pred['score'])
    
    return ret


def compute_iou(pred_boxes, gt_boxes):
    '''
    x, y, w, h
    ious: 2D array storing IoUs of all pairs of predictions and groundtruths
    '''
    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        xp, yp, wp, hp = pred
        pred_area = wp*hp
        for j, gt in enumerate(gt_boxes):
            xg, yg, wg, hg = gt
            gt_area = wg*hg

            #Width of intersection
            wi = min(xp + wp, xg + wg) - max(xp, xg)
            if wi <= 0:
                continue

            #Height of intersection
            hi = min(yp + hp, yg + hg) - max(yp, yg)
            if hi <= 0:
                continue
            intersection_area = wi * hi
            union_area = pred_area+gt_area-intersection_area

            ious[i, j] = min(1, intersection_area / union_area * 1.0)

    return ious


def get_prediction_dict(pred_json, category_id_list=[0,1,2,3,4,5]):
    '''
    convert
    {'bbox': [...],
    'category_id': 1,
    'image_id': 1,
    'score': 0.8232898712158203}
    to pred_dict_by_img_id[1]
    {'bbox': [[], ...],
    'category_id': [1, 1, 4, 4, 5],
    'score': [...}

    Param:
    pred_json: predictions loaded from output json file
    category_id_list: a list of categories to filter

    Return a dictionary for prediction results
    predict_dict = {
        image_id: {
        'bbox'       : [[bounding box coordinate 1], ...],
        'category_id': [catid1, catid2, ...],
        'score'      : [score1, score2, ...]
        }
    }
    '''
    predict_dict = {}
    for pred in pred_json:
        if pred['category_id'] in category_id_list:
            key = pred['image_id']
            if key not in predict_dict:
                predict_dict[key] = {'bbox': [pred['bbox']], 'category_id': [pred['category_id']], 'score': [pred['score']]}
            else:
                predict_dict[key]['bbox'].append(pred['bbox'])
                predict_dict[key]['category_id'].append(pred['category_id'])
                predict_dict[key]['score'].append(pred['score'])
    return predict_dict


def get_gt_dict(annotations, category_id_list=[0,1,2,3,4,5]):
    gt_dict = {}
    count = 0
    for gt in annotations:
        # if gt['category_id'] in category_id_list:
        key = gt['image_id']
        if key not in gt_dict:
            count +=1
            gt_dict[key] = {'bbox': [gt['bbox']], 'category_id': [gt['category_id']]}
        else:
            gt_dict[key]['bbox'].append(gt['bbox'])
            gt_dict[key]['category_id'].append(gt['category_id'])
    
    return gt_dict

id2cat = {}
for cat in labels['categories']:
    id2cat[cat['id']] = cat['name']
id2cat = {0: 'C', 1: 'NH', 2: 'HL', 3: 'CD', 4: 'CL', 5: 'DGT'}
gt_dict = get_gt_dict(labels['annotations'])
pred_dict = get_prediction_dict(pred)
for key in pred_dict:
    pred_dict[key]['labels'] = [id2cat[x] for x in pred_dict[key]['category_id']]
for key in gt_dict:
    gt_dict[key]['labels'] = [id2cat[x] for x in gt_dict[key]['category_id']]

def filter_bbox(img_id: int,
                 pred_dict: dict=pred_dict, gt_dict: dict=gt_dict,
                 conf_thres: float=0.0, iou_thres: float=0.5):
    tp_idx, fp_idx = [], []
    ious = compute_iou(pred_dict[img_id]['bbox'], gt_dict[img_id]['bbox'])
    # make all the elemant zero except max element in each row
    ious = ious * (ious >= np.sort(ious, axis=0)[[-1],:])
    ious_idx = np.argwhere(ious > iou_thres)
    # k (pred_id, gt_id)
    for k in ious_idx:
        if (pred_dict[img_id]['score'][k[0]] > conf_thres):
            if (pred_dict[img_id]['category_id'][k[0]] != gt_dict[img_id]['category_id'][k[1]]):
                fp_idx.append((k[0], k[1], ious[k[0]][k[1]]))
            else:
                tp_idx.append((k[0], k[1], ious[k[0]][k[1]]))
    ious = ious * (ious >= np.sort(ious, axis=0)[[-1],:])
    no_gt_idx = list(np.where(ious.sum(axis=1) == 0)[0])
    no_gt_idx = [idx for idx in no_gt_idx if pred_dict[img_id]['score'][idx] > conf_thres]
    no_pred_idx = list(np.where(ious.sum(axis=0) == 0)[0])
    n_fail = len(fp_idx) + len(no_gt_idx) + len(no_pred_idx)
    return n_fail, tp_idx, fp_idx, no_gt_idx, no_pred_idx


def draw_bbox_wo_iou(drawer, img_anno, idx_list, name):
    for idx in idx_list:
        # Draw prediction bbox
        x1, y1, w, h = img_anno['bbox'][idx]
        x2, y2 = x1 + w, y1 + h
        drawer.rectangle([(x1, y1), (x2, y2)], outline='red', width=3)
        # Draw tag
        text = name
        font = ImageFont.truetype("FiraCode-VariableFont_wght.ttf", min(30, int(1.2 * w)))
        bg_w, bg_h = font.getsize(text)
        color = 'yellow' if name == 'FN' else 'blue'
        drawer.rectangle((x1, y1 - bg_h, x1 + bg_w, y1), fill=color, outline='white')
        drawer.text((x1, y1 - bg_h), text, (0,0,0), font=font)
        # Draw category
        font = ImageFont.truetype("FiraCode-VariableFont_wght.ttf", min(20, int(w)))
        text = img_anno['labels'][idx]
        color = (0,0,0)
        if name == 'FP':
            text += ': ' + str(np.round(img_anno['score'][idx], 2))
            color = (0,0,255)
        bg_w, bg_h = font.getsize(text)
        drawer.rectangle((x1, y1 + h, x1 + bg_w, y1 + bg_h + h), fill='white')
        drawer.text((x1, y1 + h), text, color, font=font)
    return drawer


def draw_bbox_w_iou(drawer, pred_anno, gt_anno, idx_tuple, name):
    for idx in idx_tuple:
        # Draw prediction bbox
        if name == 'TP':
            x1, y1, w, h = gt_anno['bbox'][idx[1]]
        else:
            x1, y1, w, h = pred_anno['bbox'][idx[0]]
        x2, y2 = x1 + w, y1 + h
        outline = 'lime' if name == 'TP' else 'red'
        drawer.rectangle([(x1, y1), (x2, y2)], outline=outline, width=3)
        if name == 'FP':
            # Draw ground-truth cat
            font = ImageFont.truetype("FiraCode-VariableFont_wght.ttf", min(25, int(1.2 * w)))
            text = 'GT: ' + gt_anno['labels'][idx[1]]
            bg_w, bg_h = font.getsize(text)
            drawer.rectangle((x1, y1 - bg_h, x1 + bg_w, y1), fill='blue')
            drawer.text((x1, y1 - bg_h), text, (0,0,0), font=font)
        # Draw ground-truth bbox
        # x1, y1, w, h = gt_dict[img_id]['bbox'][idx[1]]
        # x2, y2 = x1 + w, y1 + h
        # drawer.rectangle([(x1, y1), (x2, y2)], outline='green', width=3)
        # Draw pred cat + score
        font = ImageFont.truetype("FiraCode-VariableFont_wght.ttf", min(20, int(w)))
        text = pred_anno['labels'][idx[0]] + ': ' + str(np.round(pred_anno['score'][idx[0]], 2))
        bg_w, bg_h = font.getsize(text)
        drawer.rectangle((x1, y1 + h, x1 + bg_w, y1 + bg_h + h), fill='white')
        drawer.text((x1, y1 + h), text, (0,0,255), font=font)
    return drawer


def bbox_visualize(img_id: int, img_root: str='test_images/'):
    _, tp_idx, fp_idx, no_gt_idx, no_pred_idx = filter_bbox(img_id)
    img = Image.open(img_root + id2fname[img_id])
    drawer = ImageDraw.Draw(img)
    ## Draw prediction without ground-truth
    drawer = draw_bbox_wo_iou(drawer, pred_dict[img_id], no_gt_idx, 'FP')
    ## Draw ground-truth without prediction
    drawer = draw_bbox_wo_iou(drawer, gt_dict[img_id], no_pred_idx, 'FN')
    ## Draw false cat
    draw_bbox_w_iou(drawer, pred_dict[img_id], gt_dict[img_id], fp_idx, 'FP')
    ## Draw TP
    draw_bbox_w_iou(drawer, pred_dict[img_id], gt_dict[img_id], tp_idx, 'TP')
    
    return img