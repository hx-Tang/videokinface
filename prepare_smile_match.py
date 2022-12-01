import random
import numpy as np
import pandas as pd
from itertools import product

data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'
subject_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/Subject_Details.txt'
file_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/File_Details.txt'
kin_label = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/Kinship_Labels.txt'

# discard_attr = ['granddaughter,grandmother', 'granddaughter,grandfather', 'grandfather,grandson']
#
# subsets2attr = {'B-B': [['brother','brother']], 'S-B': [['sister','brother'], ['brother','sister']], 'S-S': [['sister','sister']],
#                 'F-D': [['father','daughter'], ['daughter','father']], 'F-S': [['father','son'], ['son','father']],
#                 'M-S': [['mother','son'], ['son','mother']], 'M-D': [['mother','daughter'], ['daughter','mother']]}

# subsets2attr = {'F-D': [['father','daughter'], ['daughter','father']], 'F-S': [['father','son'], ['son','father']],
#                 'M-S': [['mother','son'], ['son','mother']], 'M-D': [['mother','daughter'], ['daughter','mother']]}

subsets2attr = {'F-D': [['father','daughter'], ['daughter','father']]}

subjects = pd.read_csv(subject_detail, sep='\t')
files = pd.read_csv(file_detail, sep='\t')
kin_label = pd.read_csv(kin_label, sep=',')

print(subjects)
print(files)

# filter smile type
files = files.where((files['smile spontenity'] == 'spontaneous')).dropna().astype({'subject code':'int32', 'age':'int32'})

print(files)

# load video features
video_features = {}
for index, row in files.iterrows():
    filename = row['filename']
    detail_path = data_path + '/' + filename[:-4] + '.csv'
    detail = pd.read_csv(detail_path)
    video_feature = detail[' AU12_r'].values
    video_feature = np.lib.stride_tricks.sliding_window_view(video_feature, 5)
    video_features[filename] = video_feature

print(len(video_features))

# shuffle kin_label
kin_label = kin_label.sample(frac=1).reset_index(drop=True)
print(kin_label)

subj_1_code = []
subj_2_code = []
kin_type = []

for index, row in kin_label.iterrows():
    for key, attr in subsets2attr.items():
        if [row['subj_1_kinship'],row['subj_2_kinship']] == attr[0]:
            subj_1_code.append(row['subj_1_code'])
            subj_2_code.append(row['subj_2_code'])
            kin_type.append(key)
        elif len(attr)>1:
            if [row['subj_1_kinship'],row['subj_2_kinship']] == attr[1]:
                subj_1_code.append(row['subj_2_code'])
                subj_2_code.append(row['subj_1_code'])
                kin_type.append(key)

labels = pd.DataFrame(data={'subj_1_code':subj_1_code, 'subj_2_code':subj_2_code, 'kin_type':kin_type})
labels = labels.sample(frac=1).reset_index(drop=True)
print(labels)

# train_split = labels.iloc[:int(len(labels)*0.8)]
train_split = labels.iloc[:-2]
print(train_split)
# val_split = labels.iloc[int(len(labels)*0.6):int(len(labels)*0.8)]
# print(val_split)
# test_split = labels.iloc[int(len(labels)*0.8):]
test_split = labels.iloc[-2:]
print(test_split)


def get_subjects(split):
    subjects = set([i for i in split['subj_1_code']]+[i for i in split['subj_2_code']])
    return subjects

train_subjects = get_subjects(train_split)
# val_subjects = get_subjects(val_split)
test_subjects = get_subjects(test_split)

print(train_subjects)
# print(val_subjects)
print(test_subjects)


def pair_negative(split, neg_subjects):
    neg_codes = subjects.where(subjects['subject code'].isin(neg_subjects)).dropna().astype({'subject code':'int32', 'age':'int32'})
    neg_list = []
    for index, row in split.iterrows():
        pos = subjects.loc[subjects['subject code']==row['subj_2_code']]
        negs = neg_codes.where((neg_codes['subject code'] != pos['subject code'].iloc[0]) & (neg_codes['gender'] == pos['gender'].iloc[0]) & (abs(neg_codes['age']-pos['age'].iloc[0]) < 10))['subject code'].dropna().astype({'subject code':'int32'}).values.tolist()
        # negs = neg_codes.where(
        #     (neg_codes['gender'] != pos['subject code'].iloc[0]))['subject code'].dropna().astype(
        #     {'subject code': 'int32'}).values.tolist()
        neg_list.append(negs)
    return neg_list


test_neg = pair_negative(test_split, test_subjects)
# val_neg = pair_negative(val_split, val_subjects.difference(test_subjects))
# train_neg = pair_negative(train_split, train_subjects.difference(val_subjects).difference(test_subjects))
train_neg = pair_negative(train_split, train_subjects.difference(test_subjects))

print(test_neg)
# print(val_neg)
print(train_neg)


def cos_similar(a, b):
    dot = a * b
    a_len = np.linalg.norm(a, axis=0)
    b_len = np.linalg.norm(b, axis=0)
    cos = dot.sum(axis=0) / (a_len * b_len)
    return cos


def search(query_vec,docs_vec, reverse=False):
    candidates = []
    for i in range(docs_vec.shape[0]):
        if sum(query_vec)*sum(docs_vec[i]) == 0:
            continue
        cos = cos_similar(query_vec,docs_vec[i])
        candidates.append((i,cos))
    candidates = sorted(candidates,key=lambda x:-x[1], reverse=reverse)
    if len(candidates)==0:
        return -1
    return candidates[0][0]


def pair_triplets(split, negs):
    triplets = []
    for idx, (i, row) in enumerate(split.iterrows()):
        anchor = row['subj_1_code']
        anchor = files.loc[files['subject code'] == anchor]['filename'].iloc[0]
        anchor_feature = video_features[anchor]
        pos = row['subj_2_code']
        try:
            pos = files.loc[files['subject code'] == pos]['filename'].iloc[0]
        except:
            continue
        pos_feature = video_features[pos]
        for i in range(len(anchor_feature)):
            a_feature = anchor_feature[i]
            p_match_idx = search(a_feature, pos_feature)
            p_unmatch_idx = search(a_feature, pos_feature, True)
            if p_match_idx == -1:
                continue
            p_match_idx+=2
            neg = random.choice(negs[idx])
            try:
                neg = files.loc[files['subject code'] == neg]['filename'].iloc[0]
            except:
                continue
            neg_feature = video_features[neg]
            n_match_idx = search(a_feature, neg_feature)
            if n_match_idx == -1:
                continue
            n_match_idx+=2
            triplets.append([anchor, i, pos, p_match_idx, p_unmatch_idx,neg, n_match_idx])
    return triplets


trainset = pair_triplets(train_split, train_neg)
random.shuffle(trainset)
print(trainset)
print(len(trainset))
np.save('train.npy', trainset)
# valset = pair_triplets(val_split, val_neg)
# random.shuffle(valset)
# print(valset)
# print(len(valset))
# np.save('val.npy', valset)
testset = pair_triplets(test_split, test_neg)
random.shuffle(testset)
print(testset)
print(len(testset))
np.save('test.npy', testset)