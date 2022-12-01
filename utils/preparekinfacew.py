import os
import random
import numpy as np


def load_files(path):
    '''
    Input: path to parent child folder
    Output: paths to parent files, paths to child files
    '''
    parent_files = []
    child_files = []
    for file in os.listdir(path):
        if file == "Thumbs.db":
            continue
        if file.split("_")[2][0] == "1":
            parent_files.append(os.path.join(path, file))
        elif file.split("_")[2][0] == "2":
            child_files.append(os.path.join(path, file))
    return parent_files, child_files

def make_pair_dict(ROOT_DIR):
        pairs = os.listdir(ROOT_DIR)

        parent_dict = {}
        child_dict = {}

        comb_key = 1
        for pair in pairs:
            pair_path = os.path.join(ROOT_DIR, pair)
            parent_dict[comb_key], child_dict[comb_key] = load_files(pair_path)
            comb_key += 1
        return parent_dict, child_dict


TEST_ROOT_DIR = 'D:\文档\硕士\Thesis\KinFaceW-I\KinFaceW-I\images'
TRAIN_ROOT_DIR = 'D:\文档\硕士\Thesis\KinFaceW-II\KinFaceW-II\images'


train_parent_dict, train_child_dict = make_pair_dict(TRAIN_ROOT_DIR)
test_parent_dict, test_child_dict = make_pair_dict(TEST_ROOT_DIR)


def make_pairs(parent_dict, child_dict):
    matched_pairs = []
    unmatched_pairs = []
    for parent_key in parent_dict.keys():
        for child_key in child_dict.keys():
            if parent_key == child_key:
                for file in zip(parent_dict[parent_key], child_dict[child_key]):
                    matched_pairs.append(list(file)+[1])
            else:
                for file in zip(parent_dict[parent_key], child_dict[child_key]):
                    unmatched_pairs.append(list(file)+[0])
    return matched_pairs, unmatched_pairs


train_matched_pairs, train_unmatched_pairs = make_pairs(train_parent_dict, train_child_dict)
test_matched_pairs, test_unmatched_pairs = make_pairs(test_parent_dict, test_child_dict)

random.shuffle(train_unmatched_pairs)
random.shuffle(test_unmatched_pairs)

train_pair = train_matched_pairs + train_unmatched_pairs[:len(train_matched_pairs)]
random.shuffle(train_pair)

test_pair = test_matched_pairs + test_unmatched_pairs[:len(test_matched_pairs)]
random.shuffle(test_pair)

print(train_pair, test_pair)

np.save('kfw_train_pair.npy', train_pair)
np.save('kfw_test_pair.npy', test_pair)