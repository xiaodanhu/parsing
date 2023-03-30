import collections.abc
import gzip
import dataloaderBG_FA as dataloader
import torch	
import numpy as np	
import os
import pandas as pd
import torch.nn.functional as F
import json
import math
import glob

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = [self.label]


        # while len(tree.children) == 1 and isinstance(
        #         tree.children[0], InternalTreebankNode):
        #     tree = tree.children[0]
        #     sublabels.append(tree.label)


        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)


class LeafTreebankNode_alt(TreebankNode):
    def __init__(self, tag, I3D):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(I3D, torch.Tensor)
        self.I3D = I3D

    def linearize(self):
        return "({} {})".format(self.tag, 'frame')

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode_alt(index, self.tag, self.I3D)


class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        #assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert_original(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode_alt) :
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        sp_enclosing = enclosing.oracle_splits(enclosing.left,enclosing.right)
        sp_enclosing.append(enclosing.left)
        sp_enclosing.append(enclosing.right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        elif (left in sp_enclosing) and (right in sp_enclosing):
            if enclosing.label[-1] in ['sport exercise','relaxing','personal care','household activity', '']:
                return 'BR_PHARASE_LEVEL'
            else:
                try:
                    int(enclosing.label[-1])
                except ValueError:
                    return 'BR_ACTION_LEVEL'
                else:
                    return 'BR_FRAME_LEVEL'
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]


class LeafParseNode_alt(ParseNode):
    def __init__(self, index, tag, I3D):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(I3D, torch.Tensor)
        self.I3D = I3D

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode_alt(self.tag, self.I3D)

def load_trees_wBG(mode, path):
    loaders = dataloader.construct_dataloaders(path, path+'/I3D_features', 32, 0, 44739242,3,10,10,20)
    with open(os.path.join(path, "annotations_gt.json"), 'r') as f:
        data = json.load(f)['database']
    
    trees=[]
    a=0
    assert(mode=='train' or mode=='test')
    id2activitylabel=loaders['train'].dataset.id2activitylabel
    id2phraselabel=loaders['train'].dataset.id2phraselabel
    id2actionlabel=loaders['train'].dataset.id2actionlabel
    I3D_interval = 16
    anno_interval = 8
    for record in loaders[mode].dataset:
        a+=1
        feature_folder=path+'/I3D_features'
        # n_frame = data[record[4]]['frame_num']
        # vid_feature = torch.from_numpy(pd.read_csv(os.path.join(feature_folder, record[4] + '.csv')).values).float() # .sample(n=10)
        # vid_feature = F.interpolate(vid_feature.unsqueeze(0).unsqueeze(0), [math.ceil(n_frame/I3D_interval), 2048], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        vid_feature = torch.tensor([np.load(fname)[0] for fname in sorted(glob.glob(feature_folder + '/' + record[4] + '/' + '*.npy'))])
        # if a>600:
        #     break
        # f_acts = [int(i/I3D_interval) for i in record[3].flatten() if i != 0] 
        #a_st=f_acts[0]
        #a_ed=f_acts[-1]
        activities=record[1]['activity']
        activities=[h for h in activities if h!=-1]
        ch_vdo=[]
        for h in range(len(activities)):
            if activities[h] <= 0 or activities[h] > len(id2activitylabel):
                continue
            ch_act=[]
            phs=[i for i in record[1]['phrase'][h] if i !=-1]
            #last_ph=a_st
            for i in range(len(phs)):
                ph=int(phs[i])
                ch_ph=[]
                ans=[j for j in record[1]['action'][h,i] if j !=-1]
                for j in range(len(ans)):
                    an=int(ans[j])
                    ch_an=[]
                    frs=[k for k in record[2][h,i,j] if k!=0]
                    for k in range(len(frs)):
                        I3D=record[0][h,i,j,k]
                        ch_an.append(LeafTreebankNode_alt(id2actionlabel[an], I3D))
                    if(len(ch_an)!=0):
                        ch_ph.append(InternalTreebankNode(id2actionlabel[an], ch_an))

                    if j+1<len(ans):
                        if int(max(record[3][h,i,j])/I3D_interval)+1 < int(record[3][h,i,j+1,0]/I3D_interval):
                            an_bgs=[]
                            for k in range(int(max(record[3][h,i,j])/I3D_interval)+1,int(record[3][h,i,j+1,0]/I3D_interval)):
                                I3D=vid_feature[k]
                                an_bgs.append(LeafTreebankNode_alt('BG_Action', I3D))
                            if(len(an_bgs)!=0):
                                ch_ph.append(InternalTreebankNode('BG_Action', an_bgs))

                ch_act.append(InternalTreebankNode(id2phraselabel[ph], ch_ph))

                if i+1<len(phs):
                    curr_ed=[int(j/I3D_interval) for j in record[3][h,i].flatten() if j!=0][-1]
                    next_st=int(record[3][h,i+1,0,0]/I3D_interval)
                    if curr_ed+1 < next_st:
                        ph_bgs=[]
                        an_bgs=[]
                        for k in range(curr_ed+1,next_st):
                            I3D=vid_feature[k]
                            an_bgs.append(LeafTreebankNode_alt('BG_Action', I3D))
                        if(len(an_bgs)!=0):
                            ph_bgs.append(InternalTreebankNode('BG_Action', an_bgs))
                        if(len(ph_bgs)!=0):
                            ch_act.append(InternalTreebankNode('BG_Phrase', ph_bgs))
            # if activities[h]<=4 and activities[h]>0:
            ch_vdo.append(InternalTreebankNode(id2activitylabel[activities[h]], ch_act))
            # else:
            #     ch_vdo.append(InternalTreebankNode('UNK', ch_act))
            
            if h+1<len(activities):
                curr_ed=[int(j/I3D_interval) for j in record[3][h].flatten() if j!=0][-1]
                next_st=int(record[3][h+1,0,0,0]/I3D_interval)
                if curr_ed+1<next_st:
                    act_bgs=[]
                    ph_bgs=[]
                    an_bgs=[]
                    for k in range(curr_ed+1,next_st):
                        I3D=vid_feature[k]
                        an_bgs.append(LeafTreebankNode_alt('BG_Action', I3D))
                    if(len(an_bgs)!=0):
                        ph_bgs.append(InternalTreebankNode('BG_Action', an_bgs))
                    if(len(ph_bgs)!=0):
                        act_bgs.append(InternalTreebankNode('BG_Phrase', ph_bgs))
                    if(len(act_bgs)!=0):
                        ch_vdo.append(InternalTreebankNode('BG_Activity', act_bgs))
        if len(ch_vdo) == 0:
            aa = 1
        trees.append(InternalTreebankNode('tree', ch_vdo))
                      
    return trees
