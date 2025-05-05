import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random

class IcaAVELoader(Dataset):
    def __init__(self, args, mode='train', incremental_step=0):
        self.mode = mode
        self.args = args

        self.data_root = '../data'
        self.visual_pretrained_feature_path = os.path.join(self.data_root, 'visual_pretrained_feature_dict.npy')
        self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
        self.touch_pretrained_feature_path = os.path.join(self.data_root, 'touch_pretrained_feature_dict.npy')
        self.all_touch_pretrained_features = np.load(self.touch_pretrained_feature_path, allow_pickle=True).item()

        self.all_id_category_dict = np.load(
            os.path.join(self.data_root, 'all_id_category_dict.npy'), allow_pickle=True).item()
        self.category_encode_dict = np.load(
            os.path.join(self.data_root, 'category_encode_dict.npy'), allow_pickle=True).item()
        self.all_classId_vid_dict = np.load(
            os.path.join(self.data_root, 'all_classId_vid_dict.npy'), allow_pickle=True).item()

        if self.mode == 'train':
            self.all_id_category_dict = self.all_id_category_dict['train']
            self.all_classId_vid_dict = self.all_classId_vid_dict['train']
        elif self.mode == 'val':
            self.all_id_category_dict = self.all_id_category_dict['val']
            self.all_classId_vid_dict = self.all_classId_vid_dict['val']
        elif self.mode == 'test':
            self.all_id_category_dict = self.all_id_category_dict['test']
            self.all_classId_vid_dict = self.all_classId_vid_dict['test']
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')

        self.incremental_step = incremental_step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def set_current_step_classes(self):
        if self.mode == 'train':
            current_step_class = np.array(range(self.args.class_num_per_step * self.incremental_step,self.args.class_num_per_step * (self.incremental_step + 1)))
        else:
            current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
        return current_step_class

    def current_step_data(self):
        all_current_data_vids = []
        for class_idx in self.current_step_class:
            all_current_data_vids += self.all_classId_vid_dict[class_idx]
        return all_current_data_vids

    def set_incremental_step(self, step):
        self.incremental_step = step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def __getitem__(self, index):
        vid = self.all_current_data_vids[index]
        category = self.all_id_category_dict[vid]
        category_id = self.category_encode_dict[category]

        visual_feature = self.all_visual_pretrained_features[vid]
        touch_feature = self.all_touch_pretrained_features[vid]

        return (visual_feature, touch_feature), category_id

    def close_visual_features_h5(self):
        self.all_visual_pretrained_features.close()

    def __len__(self):
        return len(self.all_current_data_vids)


class exemplarLoader(Dataset):
    def __init__(self, args, incremental_step=0):
        self.args = args
        self.data_root = '../data'
        self.visual_pretrained_feature_path = os.path.join(self.data_root, 'visual_pretrained_feature_dict.npy')
        self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
        self.touch_pretrained_feature_path = os.path.join(self.data_root, 'touch_pretrained_feature_dict.npy')
        self.all_touch_pretrained_features = np.load(self.touch_pretrained_feature_path, allow_pickle=True).item()

        self.all_id_category_dict = np.load(os.path.join(self.data_root, 'all_id_category_dict.npy'), allow_pickle=True).item()['train']
        self.all_classId_vid_dict = np.load(os.path.join(self.data_root, 'all_classId_vid_dict.npy'), allow_pickle=True).item()['train']
        self.category_encode_dict = np.load(os.path.join(self.data_root, 'category_encode_dict.npy'), allow_pickle=True).item()

        self.incremental_step = incremental_step

        self.exemplar_class_vids_set = []
        self.exemplar_vids_set = []

    def _set_incremental_step_(self, step):
        self.incremental_step = step
        self._update_exemplars_()

    def _update_exemplars_(self):
        if self.incremental_step == 0:
            return
        new_memory_classes = range((self.incremental_step - 1) * self.args.class_num_per_step,
                                   self.incremental_step * self.args.class_num_per_step)
        exemplar_num_per_class = self.args.memory_size // (self.incremental_step * self.args.class_num_per_step)
        new_memory_class_exemplars = self._init_new_memory_class_exemplars_(new_memory_classes, exemplar_num_per_class)

        if self.incremental_step == 1:
            self.exemplar_class_vids_set += new_memory_class_exemplars
        else:
            for i in range(len(self.exemplar_class_vids_set)):
                self.exemplar_class_vids_set[i] = self.exemplar_class_vids_set[i][:exemplar_num_per_class]

            self.exemplar_class_vids_set += new_memory_class_exemplars

        self.exemplar_vids_set = np.array(self.exemplar_class_vids_set).reshape(-1).tolist()
        self.exemplar_vids_set = [vid for vid in self.exemplar_vids_set if vid is not None]

    def _init_new_memory_class_exemplars_(self, new_memory_classes, exemplar_num_per_class):
        new_memory_class_exemplars = []
        for i in new_memory_classes:
            class_vids = self.all_classId_vid_dict[i]

            class_exemplar = random.sample(class_vids, min(len(class_vids), exemplar_num_per_class))
            if len(class_vids) < exemplar_num_per_class:
                class_exemplar += [None for i in range(exemplar_num_per_class - len(class_vids))]
            new_memory_class_exemplars.append(class_exemplar)
        return new_memory_class_exemplars

    def __getitem__(self, index):
        vid = self.exemplar_vids_set[index]

        category = self.all_id_category_dict[vid]
        category_id = self.category_encode_dict[category]

        visual_feature = self.all_visual_pretrained_features[vid]
        visual_feature = torch.Tensor(visual_feature)
        touch_feature = self.all_touch_pretrained_features[vid]
        touch_feature = torch.Tensor(touch_feature)

        return (visual_feature, touch_feature), category_id

    def close_visual_features_h5(self):
        self.all_visual_pretrained_features.close()

    def __len__(self):
        return len(self.exemplar_vids_set)
