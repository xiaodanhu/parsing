import collections.abc
import gzip
import dataloaderorg
import torch

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

        
        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)
        

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
            if enclosing.label[-1] in ['VT','FX','UB','BB']:
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


def load_trees_alt(mode, path='/data/xiaodan8/research/dataset/FineGym'):
    #with open(path) as infile:
    #    treebank = infile.read()
    loaders = dataloaderorg.construct_dataloaders(path, path+'/I3D_features', 32, 0, 44739242,1,5,5,20)
    trees=[]
    i=0
    assert(mode=='train' or mode=='test')
    id2activitylabel=loaders['train'].dataset.id2activitylabel
    id2phraselabel=loaders['train'].dataset.id2phraselabel
    id2actionlabel=loaders['train'].dataset.id2actionlabel
    #print(id2actionlabel[40])
    for record in loaders[mode].dataset:
        
        childrenact=[]
        #print(type(record))
        # labels: snippet_fts, labels, record.masks, og_locations
        act=record[1]['activity']
        for phrase in range(len(record[1]['phrase'][0])):
            if (record[1]['phrase'][0,phrase]==-1):
                break
            childrenph=[]
            ph=int(record[1]['phrase'][0,phrase])
            for action in range(len(record[1]['action'][0,phrase])):
                if (record[1]['action'][0,phrase,action]==-1):
                    break
                childrenan=[]
                an=int(record[1]['action'][0,phrase,action])
                #print(an)
                for frame in range(len(record[2][0,phrase,action])):
                    if (record[2][0,phrase,action,frame]==0):
                        break
                    I3D=record[0][0,phrase,action,frame]
                    #print(i,phrase,action,childrenan)
                    childrenan.append(LeafTreebankNode_alt(id2actionlabel[an], I3D))
                #    print(phrase,action,frame,mode)
                #    if(i==616):
                #        print(childrenan)
                #if(i==616):
                #    print(phrase,action,childrenan)
                if(len(childrenan)!=0):
                    childrenph.append(InternalTreebankNode(id2actionlabel[an], childrenan))
            if(len(childrenph)!=0):
                childrenact.append(InternalTreebankNode(id2phraselabel[ph], childrenph))
        if act<=4 and act>=0:
            trees.append(InternalTreebankNode(id2activitylabel[act], childrenact))
        else:
            trees.append(InternalTreebankNode('UNK', childrenact))
        #i+=1
        #print(i)

    return trees

def build_tree(loaders,mode):
    #with open(path) as infile:
    #    treebank = infile.read()
    #loaders = dataloaderorg.construct_dataloaders(path, path+'/I3D_features', 32, 0, 44739242,1,5,5,20)
    trees=[]
    i=0
    assert(mode=='train' or mode=='test')
    id2activitylabel=loaders['train'].dataset.id2activitylabel
    id2phraselabel=loaders['train'].dataset.id2phraselabel
    id2actionlabel=loaders['train'].dataset.id2actionlabel
    #print(id2actionlabel[40])
    for record in loaders[mode].dataset:
        
        childrenact=[]
        #print(type(record))
        # labels: snippet_fts, labels, record.masks, og_locations
        act=record[1]['activity']
        for phrase in range(len(record[1]['phrase'][0])):
            if (record[1]['phrase'][0,phrase]==-1):
                break
            childrenph=[]
            ph=int(record[1]['phrase'][0,phrase])
            for action in range(len(record[1]['action'][0,phrase])):
                if (record[1]['action'][0,phrase,action]==-1):
                    break
                childrenan=[]
                an=int(record[1]['action'][0,phrase,action])
                #print(an)
                for frame in range(len(record[2][0,phrase,action])):
                    if (record[2][0,phrase,action,frame]==0):
                        break
                    I3D=record[0][0,phrase,action,frame]
                    #print(i,phrase,action,childrenan)
                    childrenan.append(LeafTreebankNode_alt(id2actionlabel[an], I3D))
                #    print(phrase,action,frame,mode)
                #    if(i==616):
                #        print(childrenan)
                #if(i==616):
                #    print(phrase,action,childrenan)
                if(len(childrenan)!=0):
                    childrenph.append(InternalTreebankNode(id2actionlabel[an], childrenan))
            if(len(childrenph)!=0):
                childrenact.append(InternalTreebankNode(id2phraselabel[ph], childrenph))
        if act<=4 and act>=0:
            trees.append(InternalTreebankNode(id2activitylabel[act], childrenact))
        else:
            trees.append(InternalTreebankNode('UNK', childrenact))
        #i+=1
        #print(i)

    return trees