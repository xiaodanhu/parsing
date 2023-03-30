import torch
import os
import json
import cv2
import functools
import numpy as np
import math
import argparse
from pathlib import Path
import glob

class Record:
    def __init__(self, vid, locations, masks, fps, interval, window_size, activity_id, phrase_id, action_id, s_len, action_label, phrase_label):
        self.vid = vid
        self.locations = locations
        self.masks = masks
        self.s_len = s_len
        self.interval = interval
        self.window_size = window_size
        self.base = self.locations[..., 0]
        _, dim1, dim2 = np.shape(self.base)
        self.locations_norm = np.zeros(np.shape(self.locations))
        for i in range(dim1):
            for j in range(dim2):
                self.locations_norm[0, i, j, :] = (self.locations[0, i, j, :] - self.base[0, i, j]) / (self.window_size * self.interval)
        self.locations_norm = np.multiply(self.locations_norm, self.masks) + 0
        self.fps = fps
        self.activity_id = activity_id
        self.phrase_id = phrase_id
        self.action_id = action_id
        self.action_label = action_label
        self.phrase_label = phrase_label


class CubeDataset(torch.utils.data.Dataset):
    """
    Construct CubeDataset
    """
    def __init__(self, root: str, feature_folder:str, mode: str, max_activity_len_per_record: int, max_phrase_len_per_activity: int, max_action_len_per_phrase: int, max_movment_len_per_action: int) -> None:
        super(CubeDataset, self).__init__()
        
        self.max_activity_len_per_record = max_activity_len_per_record
        self.max_phrase_len_per_activity = max_phrase_len_per_activity
        self.max_action_len_per_phrase = max_action_len_per_phrase
        self.max_movment_len_per_action = max_movment_len_per_action
        self.interval = 8

        self.feature_folder = feature_folder

        self.activitylabel2id, self.phraselabel2id, self.actionlabel2id, self.phrase2activity, self.action2phr, self.vid_info = self.get_index(root, mode)

        self.n_activitylabel = max(self.activitylabel2id.values()) + 1 #len(self.activitylabel2id)
        self.n_phraselabel = max(self.phraselabel2id.values()) + 1 #len(self.phraselabel2id)
        self.n_actionlabel = max(self.actionlabel2id.values()) + 1 #len(self.actionlabel2id)

        self.records, self.samples, self.masks, self.activity_label_list, self.phrase_label_list, self.action_label_list, self.s_e_action, self.locations = self.load_file(root, mode)

        self.id2activitylabel = {v:k for k,v in self.activitylabel2id.items()}
        self.id2phraselabel = {v:k for k,v in self.phraselabel2id.items()}
        self.id2actionlabel = {v:k for k,v in self.actionlabel2id.items()}


    def second2frame(self, second, fps):
        frame = int(second * fps)
        return frame

    def get_index(self, root:str, mode:str):
        '''
        load annotations from json and txt files
        '''
        anno_path = os.path.join(root, "valid_anno.json")
        # activitylabel2id = {"Floor Exercise": 0, "Balance Beam": 1, "Uneven Bars": 2, "Vault-Women": 3}
        activitylabel2id = {"FX": 2, "BB": 3, "UB": 4, "VT": 1}
        phrase_label_path = os.path.join(root, "set_categories.txt")
        phraselabel2id = {}
        phrase2activity = {}
        with open(phrase_label_path, 'r') as f:
            for line in f:
                line = line.strip().split("; ")
                phrase_id = int(line[0].strip("set:  "))
                activity_id = activitylabel2id[line[1][:2]]
                if len(phrase2activity.keys()) == 0:
                    phrase2activity = {phrase_id: activity_id}
                else:
                    phrase2activity.update({phrase_id: activity_id})

                phrase = line[1]
                if len(phraselabel2id.keys()) == 0:
                    phraselabel2id = {phrase: phrase_id}
                else:
                    phraselabel2id.update({phrase: phrase_id})

        action_label_path = os.path.join(root, "gym99_categories.txt")
        actionlabel2id = {}
        action2phr = {}
        with open(action_label_path, 'r') as f:
            for line in f:
                line = line.strip().split("; ")
                act_id = int(line[0].strip("Clabel:   "))
                phr_id = int(line[1].strip("set:  "))
                if len(action2phr.keys()) == 0:
                    action2phr = {act_id: phr_id}
                else:
                    action2phr.update({act_id: phr_id})

                action = line[2][8:]
                if len(actionlabel2id.keys()) == 0:
                    actionlabel2id = {action: act_id}
                else:
                    actionlabel2id.update({action: act_id})


        with open(anno_path, 'r') as f:
            data = json.load(f)

        video_list = list(data.keys())
        vid_info = {}
        for vid in video_list:
            v = cv2.VideoCapture(root + '/video/gym_' + vid + '.mp4')
            vid_info[vid] = {"fps": v.get(cv2.CAP_PROP_FPS), "w": v.get(cv2.CAP_PROP_FRAME_WIDTH), "h": v.get(cv2.CAP_PROP_FRAME_HEIGHT), "duration":int(v.get(cv2.CAP_PROP_FRAME_COUNT))}

        return activitylabel2id, phraselabel2id, actionlabel2id, \
               phrase2activity, action2phr, vid_info

    def load_file(self, root:str, mode:str):

        anno_label_path = os.path.join(root, "gym99_merged_" + f"{mode}.txt")
        anno_path = os.path.join(root, "valid_anno.json")
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        action_label_list = []
        phrase_label_list = []
        activity_label_list = []
        s_e_activity = []
        s_e_action = []
        locations = []
        sample = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
        samples = []
        mask = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
        masks = []
        s_len = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase))
        records = []
        action_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase)) - 1
        phrase_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity)) - 1
        ct_actv = 0
        ct_phra = 0
        ct_actn = 0
        vname_old=''
        action_id_old=0
        phrase_id_old=0
        activity_id_old=0
        flag_save = False
        with open(anno_label_path, 'r') as f:
            for line in f:
                line, action_id = line.strip().split(" ")
                action_id = int(action_id)
                line = line.strip().split("_")
                vname = line[0]
                activity_key = "_".join(line[1:4])
                action_key = "_".join(line[4:7])

                if vname in data.keys() and \
                    activity_key in data[vname] and \
                    action_key in data[vname][activity_key]['segments'] and \
                    'I3D_' + vname + '.npy' in os.listdir(self.feature_folder):

                    phrase_id = self.action2phr[action_id]
                    activity_data = data[vname][activity_key]
                    activity_id = activity_data['event']

                    # count
                    if len(activity_label_list) > 0:
                        if activity_id != activity_label_list[-1] or vname != vname_old:
                            ct_phra = 0
                            ct_actn = 0
                            if ct_actv < self.max_activity_len_per_record - 1:
                                ct_actv += 1
                            else:
                                flag_save = True
                                # samples.append(sample)
                                # masks.append(mask)
                                # records.append(Record(vname_old, sample, mask, self.vid_info[vname_old]['fps'], self.interval, self.max_movment_len_per_action, activity_id_old, phrase_id_old, action_id_old, s_len, action_label, phrase_label))
                                # sample = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
                                # mask = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
                                # s_len = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase))
                                # action_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase)) - 1
                                # phrase_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity)) - 1
                                # ct_actv = 0
                        else:
                            if phrase_id != phrase_label_list[-1]:
                                ct_actn = 0
                                if ct_phra < self.max_phrase_len_per_activity - 1:
                                    ct_phra += 1
                                else:
                                    ct_phra = 0
                                    ct_actv += 1
                                    if ct_actv > self.max_activity_len_per_record - 1:
                                        flag_save = True
                                    # continue
                            else:
                                if action_id != action_label_list[-1]:
                                    if ct_actn < self.max_action_len_per_phrase - 1:
                                        ct_actn += 1
                                    else:
                                        ct_actn = 0
                                        ct_phra += 1
                                        if ct_phra > self.max_phrase_len_per_activity - 1 and ct_actv >= self.max_activity_len_per_record - 1:
                                            flag_save = True
                                        # continue
                    
                    if flag_save:
                        samples.append(sample)
                        masks.append(mask)
                        records.append(Record(vname_old, sample, mask, self.vid_info[vname_old]['fps'], self.interval, self.max_movment_len_per_action, activity_id_old, phrase_id_old, action_id_old, s_len, action_label, phrase_label))
                        sample = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
                        mask = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase, self.max_movment_len_per_action))
                        s_len = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase))
                        action_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity, self.max_action_len_per_phrase)) - 1
                        phrase_label = np.zeros((self.max_activity_len_per_record, self.max_phrase_len_per_activity)) - 1
                        ct_actv, ct_phra, ct_actn = 0, 0, 0
                        flag_save = False


                    activity_label_list.append(activity_id)
                    phrase_label_list.append(phrase_id)
                    action_label_list.append(action_id)

                    # if vname == '1Fdwuy2V9EY':
                    #     aaa = 1
                    activity_time = list(map(functools.partial(self.second2frame, fps=self.vid_info[vname]['fps']), activity_data['timestamps'][0]))
                    s_e_activity.append(tuple(activity_time))
                    action_time = [activity_data['timestamps'][0][0]+i for i in activity_data['segments'][action_key]['timestamps'][0]]
                    action_time = list(map(functools.partial(self.second2frame, fps=self.vid_info[vname]['fps']), action_time))
                    s_e_action.append(tuple(action_time))
                    # [66521, 66569] -> [66528, 66568]
                    s = int( self.interval * math.ceil( action_time[0] / self.interval ))
                    e = int( self.interval * math.ceil( action_time[1] / self.interval ))
                    frames = np.array(range(s, e, self.interval))
                    seq_len = len(frames)
                    # [10 18 26 ... 42] 50 58
                    # [10 18 26       ]
                    if seq_len <= self.max_movment_len_per_action:
                        location_action = np.zeros((self.max_movment_len_per_action))
                        location_action[:seq_len] = frames
                    else:
                        location_action = frames[:self.max_movment_len_per_action]
                    locations.append(location_action)

                    sample[ct_actv, ct_phra, ct_actn, ...] = location_action
                    mask[ct_actv, ct_phra, ct_actn, :min(seq_len, self.max_movment_len_per_action)] = 1
                    s_len[ct_actv, ct_phra, ct_actn] = min(seq_len, self.max_movment_len_per_action)

                    action_label[ct_actv, ct_phra, ct_actn] = action_id
                    phrase_label[ct_actv, ct_phra] = phrase_id

                    vname_old=vname
                    action_id_old=action_id
                    activity_id_old=activity_id
                    phrase_id_old=phrase_id

        return records, samples, masks, activity_label_list, phrase_label_list, action_label_list, s_e_action, locations

    def get_data(self, record: Record):
        
        og_locations = torch.Tensor(record.locations)

        vid_feature = np.load(os.path.join(self.feature_folder, 'I3D_' + record.vid + '.npy'))
        
        _, dim1, dim2, dim3 = np.shape(og_locations)
        snippet_fts = torch.zeros((1, dim1, dim2, dim3, 2048))
        for i in range(dim1):
            for j in range(dim2):
                for k in range(dim3):
                    if record.masks[0, i, j, k] == 0:
                        break
                    ft_idx = min(torch.floor_divide(og_locations[0, i, j, k], 8).int(), len(vid_feature) - 1)
                    snippet_fts[0, i, j, k, :] = torch.from_numpy(vid_feature[ft_idx].squeeze())

        labels = {'activity': record.activity_id, 'phrase': record.phrase_label, 'action': record.action_label}

        return snippet_fts, labels, record.masks, og_locations, torch.from_numpy(np.array([i[0] for i in vid_feature])), record.vid

    def __getitem__(self, index):
        return self.get_data(self.records[index])

    def __len__(self):
        return len(self.records)

    def collate_fn(self, batch):
        encodings = torch.stack([instance[0] for instance in batch])
        labels_activity, labels_phrase, labels_action = zip(*[[instance[1]['activity'],instance[1]['phrase'],instance[1]['action']] for instance in batch])
        masks = torch.from_numpy(np.stack([instance[2] for instance in batch]))
        locations = torch.from_numpy(np.stack([instance[3] for instance in batch]))
        return encodings, masks, torch.tensor(labels_activity), torch.tensor(labels_phrase).flatten().type(torch.int64), torch.tensor(labels_action).flatten().type(torch.int64), locations



def construct_dataloaders(root:str, feature_folder:str, batch_size:int=4, num_workers:int=4, seed:int=44739242,
        max_activity_len_per_record:int=1, max_phrase_len_per_activity:int=10, max_action_len_per_phrase:int=10, max_movment_len_per_action:int=100):
    '''
    root: where you put train.txt, dev.txt and test.txt
    '''
    datasets = {"train": CubeDataset(root=root, feature_folder=feature_folder, mode="train", max_activity_len_per_record=max_activity_len_per_record, max_phrase_len_per_activity=max_phrase_len_per_activity, max_action_len_per_phrase=max_action_len_per_phrase, max_movment_len_per_action=max_movment_len_per_action)}
    datasets["test"] = CubeDataset(root=root, feature_folder=feature_folder, mode="test", max_activity_len_per_record=max_activity_len_per_record, max_phrase_len_per_activity=max_phrase_len_per_activity, max_action_len_per_phrase=max_action_len_per_phrase, max_movment_len_per_action=max_movment_len_per_action)
    dataloaders = {sp: torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sp == 'train',
        drop_last=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed)
        ) for sp, dataset in datasets.items()}
    return dataloaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="/hdd/xiaodan8/dataset/FineGym", help="")
    parser.add_argument('--max_activity_len_per_record', type=int, default=1)
    parser.add_argument('--max_phrase_len_per_activity', type=int, default=5)
    parser.add_argument('--max_action_len_per_phrase', type=int, default=5)
    parser.add_argument('--max_movment_len_per_action', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32, help="")
    parser.add_argument('--eval-batch-size', type=int, default=32, help="")
    parser.add_argument('--num-workers', type=int, default=0, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=str, default='2,3', help="gpu")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', type=int, default=44739242, help="random seed")
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    opts = parser.parse_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)


    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    opts.feature_folder = opts.root + '/I3D_features'
    loaders = construct_dataloaders(opts.root, opts.feature_folder, opts.batch_size, opts.num_workers, opts.seed,
        opts.max_activity_len_per_record, opts.max_phrase_len_per_activity, opts.max_action_len_per_phrase, opts.max_movment_len_per_action)
    opts.n_activitylabel = loaders["train"].dataset.n_activitylabel
    opts.n_phraselabel = loaders["train"].dataset.n_phraselabel
    opts.n_actionlabel = loaders["train"].dataset.n_actionlabel