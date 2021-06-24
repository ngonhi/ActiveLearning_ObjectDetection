import os
import numpy as np
import json

from numpy.lib.function_base import average
import utils
import cv2
import matplotlib.pyplot as plt
from itertools import combinations

NUM_CLASSES=6

def KL_divergence(p, q):
    '''
    p: numpy array of probability distribution P
    q: numpy array of probability distribution Q
    '''
    # If either p or q is 0, the value is 0
    ret = 0
    cross_entropy = 0
    entropy = 0
    for i in range(p.shape[0]):
        p_val = p[i]
        q_val = q[i]

        if q_val and p_val:
            cross_entropy -= p_val * np.log(q_val)
            entropy -= p_val * np.log(p_val)  
    ret = cross_entropy - entropy

    return ret


def KL_score(pred_dict_list):
    '''
    Calculate average confidence score distribution for each image with all models 
    Compare the score distribution between all different combination of models using KL divergence
    Take average of KL divergence
    (Haussman et. al, 2020)

    mean(KL(model1 | model2) + KL(model2 | model3) + ... + KL(modeln|model1))

    If a class does not exist in an image, its entropy is 0.

    This method is not correct as there are KL values < 0
    '''
    num_models = len(pred_dict_list)
    image_ids = list(pred_dict_list[0].keys())
    image_ids.sort()
    image_kl = np.zeros((1, len(image_ids)))
    for index, image_id in enumerate(image_ids):
        # Calculate average confidence score per class per model
        average_class_conf_score = np.zeros((num_models, NUM_CLASSES))
        for i in range(num_models):
            pred_dict = pred_dict_list[i]
            pred_img = pred_dict[image_id]
            for c in range(NUM_CLASSES):
                class_conf_score = np.array(pred_img[c])
                if len(class_conf_score):
                    average_class_conf_score[i, c] = np.mean(class_conf_score)

        # Probability distribution of confidence score for each class
        average_class_conf_score = average_class_conf_score / np.sum(average_class_conf_score, axis=1).reshape((num_models, -1))
        # Calculate KL divergence
        combs = combinations(range(num_models), 2)
        combs_len = sum(1 for _ in combs)
        KL = np.empty((1, combs_len))
        for i, comb in enumerate(combs):
            model1, model2 = comb
            KL[0, i] = KL_divergence(average_class_conf_score[model1], average_class_conf_score[model2])
        
        image_kl[0, index] = np.mean(KL)

    return image_kl


def mutual_information(pred_dict_list):
    '''
    Make use of an ensemble of models to measure disagreement
    Arrive at an uncertainty score for each image
    (Haussman et. al, 2020)

    If a class does not exist in an image, its entropy is 0.
    '''
    num_models = len(pred_dict_list)
    image_ids = list(pred_dict_list[0].keys())
    image_ids.sort()
    image_mi = np.zeros((1, len(image_ids)))

    for index, image_id in enumerate(image_ids):
        class_entropy= np.zeros((num_models, NUM_CLASSES))
        average_class_conf_score = np.zeros((num_models, NUM_CLASSES))
        for i in range(num_models):
            pred_dict = pred_dict_list[i]
            pred_img = pred_dict[image_id]
            for c in range(NUM_CLASSES):
                class_conf_score = np.array(pred_img[c])
                if len(class_conf_score):
                    class_entropy[i, c] = np.sum(-class_conf_score * np.log(class_conf_score)) # Compute entropy
                    average_class_conf_score[i,c] = np.mean(class_conf_score)
        
        average_entropy = np.mean(class_entropy, axis=0)

        average_class_conf_score[average_class_conf_score==0] = np.nan   
        temp = -average_class_conf_score * np.log(average_class_conf_score)
        temp[temp==np.nan] = 0
        temp = np.nan_to_num(temp)
        entropy_average_class_conf_score = np.sum(temp, axis=0)

        image_mi[0, index] = np.max(entropy_average_class_conf_score - average_entropy)
        # image_mi[0, index] = np.sum(entropy_average_class_conf_score - average_entropy)

    return image_mi


def max_entropy(pred_dict):
    '''
    Calculate the entropy of bounding boxes of a given class and subsequently
    uses it to determine the entropy of the entire image.
    entropy of image = maximum of entropy of all classes
    The image with maximum entropy is selected for querying.
    (Soumya et. al, 2018)

    If a class does not exist in an image, its entropy is 0.
    '''
    pred_dict = pred_dict[0]
    image_ids = list(pred_dict.keys())
    image_ids.sort()
    image_entropy = np.zeros((1, len(image_ids)))
    for index, image_id in enumerate(image_ids):
        pred_img = pred_dict[image_id]
        class_entropy = np.zeros((1,NUM_CLASSES))
        for c in range(NUM_CLASSES):
            class_conf_score = np.array(pred_img[c])
            if len(class_conf_score):
                class_entropy[0,c] = np.sum(-class_conf_score * np.log(class_conf_score)) # Compute entropy
        image_entropy[0, index] = np.max(class_entropy)

    return image_entropy


def sum_entropy(pred_dict):
    '''
    Calculate the entropy of bounding boxes of a given class and subsequently
    uses it to determine the entropy of the entire image.
    entropy of image = sum of all entropy of all classes
    The image with maximum entropy is selected for querying.
    (Soumya et. al, 2018)

    If a class does not exist in an image, its entropy is 0.
    '''
    pred_dict = pred_dict[0]
    image_ids = list(pred_dict.keys())
    image_ids.sort()
    image_entropy = np.zeros((1, len(image_ids)))
    for index, image_id in enumerate(image_ids):
        pred_img = pred_dict[image_id]
        class_entropy = np.zeros((1,NUM_CLASSES))
        for c in range(NUM_CLASSES):
            class_conf_score = np.array(pred_img[c])
            if len(class_conf_score):
                class_entropy[0,c] = np.sum(-class_conf_score * np.log(class_conf_score)) # Compute entropy
        image_entropy[0, index] = np.sum(class_entropy)

    return image_entropy

with open('../TFS_analyze/TFS_vinai_batch4/labels.json') as f:
    labels = json.load(f)

id2fname = {}
for im in labels['images']:
    id2fname[im['id']] = im['file_name']

def test():
    pred_dict = utils.create_im_dict_from_path(['../TFS_analyze/TFS_vinai_batch4/val_5k.json'])
    pred_dict_list = utils.create_im_dict_from_path(['../TFS_analyze/TFS_vinai_batch4/val_5k.json',
                                                    '../TFS_analyze/test_data_20210524/test_bbox_results.json'])
    
    image_mi = sum_entropy(pred_dict) 
    uncertain_img_index = image_mi.argsort()[0][::-1][:10]
    image_ids = list(pred_dict.keys())
    image_ids.sort()
    print(uncertain_img_index)
    '''
    for i in uncertain_img_index:
        image_id = image_ids[i]
        
        # image_file = id2fname[image_id]
        # im = cv2.cvtColor(cv2.imread(os.path.join('../TFS_analyze/test_data_20210524/data', image_file)), cv2.COLOR_BGR2RGB)
        # plt.imshow(im)
        # plt.show()
        
        img = utils.bbox_visualize(image_id, img_root='../TFS_analyze/test_data_20210524/data/')
        img.show()
    '''
if __name__ == '__main__':
    test()
