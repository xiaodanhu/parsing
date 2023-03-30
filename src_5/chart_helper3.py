import numpy as np
from numpy import ndarray

ORACLE_PRECOMPUTED_TABLE = {}
def label2top(label):
    if label in [0,1,2,3]:
        return 0
    elif label in [4,5,6]:
        return 1
    elif label in [7,8]:
        return 2
    elif label ==9:
        return 3
def decode(force_gold, n_frame, label_scores_chart, is_train, gold, label_vocab, important_node, max_branch=20):
    NEG_INF = -np.inf

    label_scores_chart_copy = label_scores_chart.copy()
    value_chart = np.zeros((n_frame+1, n_frame+1,10), dtype=np.float32)#this node at 0f 1a 2p 3av level
    split_idx_chart = np.zeros((n_frame+1, n_frame+1,4), dtype=np.int32)#child at 0f 1a 2p 3av level
    best_label_chart = -np.ones((n_frame+1, n_frame+1,10), dtype=np.int32)# comb of st -> ed

    #cdef int length
    #cdef int left
    #cdef int right

    #label_scores_for_span

    #cdef int oracle_label_index
    #cdef DTYPE_t label_score
    #cdef int argmax_label_index
    #cdef DTYPE_t left_score
    #cdef DTYPE_t right_score

    #cdef int best_split
    #cdef int split_idx # Loop variable for splitting
    #cdef DTYPE_t split_val # best so far
    #cdef DTYPE_t max_split_val

    #cdef int label_index_iter

    #cdef np.ndarray[int, ndim=2] oracle_label_chart
    #cdef np.ndarray[int, ndim=3] oracle_split_chart
    if is_train or force_gold:
        if gold not in ORACLE_PRECOMPUTED_TABLE:
            oracle_label_chart = np.zeros((n_frame+1, n_frame+1), dtype=np.int32)
            oracle_split_chart = -np.ones((n_frame+1, n_frame+1), dtype=np.int32)
            for length in range(1, n_frame + 1):
                for left in range(0, n_frame + 1 - length):
                    right = left + length
                    oracle_label_chart[left, right] = label_vocab.index(gold.oracle_label(left, right))
                    if length == 1:
                        continue
                    oracle_splits = gold.oracle_splits(left, right)
                    oracle_split_chart[left, right] = min(oracle_splits)
            if not gold.nocache:
                ORACLE_PRECOMPUTED_TABLE[gold] = oracle_label_chart, oracle_split_chart
        else:
            oracle_label_chart, oracle_split_chart = ORACLE_PRECOMPUTED_TABLE[gold]

    for length in range(1, n_frame + 1):
        for left in range(0, n_frame + 1 - length):
            right = left + length

            if is_train or force_gold:
                oracle_label_index = oracle_label_chart[left, right] 

            if force_gold:
                label_score = label_scores_chart_copy[left, right, oracle_label_index]
                best_label_chart[left, right] = oracle_label_index

            else:
                if is_train:
                    # augment: here we subtract 1 from the oracle label
                    label_scores_chart_copy[left, right, oracle_label_index] -= 1

                # We do argmax ourselves to make sure it compiles to pure C
                if length < n_frame:
                    argmax_label_index = [0,0,0,0,0,0,0,0,0,0]
                else:
                    # Not-a-span label is not allowed at the root of the tree
                    argmax_label_index = [1,1,1,1,1,1,1,1,1,1]
################
                label_score = np.zeros((10))+NEG_INF

                #'''
                #0:f->f
                argmax_label_index[0] = important_node[6]
                label_score[0]=label_scores_chart_copy[left, right, important_node[6]]
                #1:f->a
                for label_index_iter in range(important_node[3]+1, important_node[4]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[1]:
                        argmax_label_index[1] = label_index_iter
                        label_score[1] = label_scores_chart_copy[left, right, label_index_iter]
                #2:f->p
                for label_index_iter in range(important_node[4], important_node[5]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[2]:
                        argmax_label_index[2] = label_index_iter
                        label_score[2] = label_scores_chart_copy[left, right, label_index_iter]
                #3:f->av
                for label_index_iter in range(important_node[5], important_node[6]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[3]:
                        argmax_label_index[3] = label_index_iter
                        label_score[3] = label_scores_chart_copy[left, right, label_index_iter]
                #4:a->a
                argmax_label_index[4] = important_node[3]
                label_score[4]=label_scores_chart_copy[left, right, important_node[3]]
                #5:a->p
                for label_index_iter in range(important_node[1]+1, important_node[2]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[5]:
                        argmax_label_index[5] = label_index_iter
                        label_score[5] = label_scores_chart_copy[left, right, label_index_iter]
                #6:a->av
                for label_index_iter in range(important_node[2], important_node[3]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[6]:
                        argmax_label_index[6] = label_index_iter
                        label_score[6] = label_scores_chart_copy[left, right, label_index_iter]
                #7:p->p
                argmax_label_index[7] = important_node[1]
                label_score[7]=label_scores_chart_copy[left, right, important_node[1]]
                #8:p->av
                for label_index_iter in range(1, important_node[1]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score[8]:
                        argmax_label_index[8] = label_index_iter
                        label_score[8] = label_scores_chart_copy[left, right, label_index_iter]
                #9:av->av
                argmax_label_index[9] = important_node[0]
                label_score[9]=label_scores_chart_copy[left, right, important_node[0]]
                #'''
                best_label_chart[left, right,:] = argmax_label_index
################
                if is_train:
                    # augment: here we add 1 to all label scores
                    label_score += 1

            if length == 1:
                value_chart[left, right] = label_score
                continue

            if force_gold:
                best_split = oracle_split_chart[left, right]#?????????????
################
            else:
                best_split =np.zeros((4))+left+1
                split_val = np.zeros((4))+NEG_INF
                for split_idx in range(left + 1, right):
                    max_split_val = np.zeros((4))+NEG_INF
                    max_split_val[0] = value_chart[left, split_idx,0] + value_chart[split_idx, right,0]
                    max_split_val[1] = max([value_chart[left, split_idx,i] + value_chart[split_idx, right,j] for i in [1,4] for j in [1,4]])
                    max_split_val[2] = max([value_chart[left, split_idx,i] + value_chart[split_idx, right,j] for i in [2,5,7] for j in [2,5,7]])
                    max_split_val[3] = max([value_chart[left, split_idx,i] + value_chart[split_idx, right,j] for i in [3,6,8,9] for j in [3,6,8,9]])
                    for i in range(4):
                        if max_split_val[i] > split_val[i]:
                            split_val[i] = max_split_val[i]
                            best_split[i] = split_idx
            if force_gold:
                value_chart[left, right] = label_score + value_chart[left, best_split] + value_chart[best_split, right] 
                split_idx_chart[left,right] = best_split        
            else:       
                value_chart[left, right,0]=label_score[0]+split_val[0]#f->f
                value_chart[left, right,1]=label_score[1]+split_val[0]#f->a
                value_chart[left, right,2]=label_score[2]+split_val[0]#f->p
                value_chart[left, right,3]=label_score[3]+split_val[0]#f->av
                value_chart[left, right,4]=label_score[4]+split_val[1]#a->a
                value_chart[left, right,5]=label_score[5]+split_val[1]#a->p
                value_chart[left, right,6]=label_score[6]+split_val[1]#a->av
                value_chart[left, right,7]=label_score[7]+split_val[2]#p->p
                value_chart[left, right,8]=label_score[8]+split_val[2]#p->av
                value_chart[left, right,9]=label_score[9]+split_val[3]#av->av
                split_idx_chart[left, right,:] = best_split

                if length==n_frame:
                    value_chart[left,right,0:3]=NEG_INF
                    value_chart[left,right,4:6]=NEG_INF
                    value_chart[left,right,7]=NEG_INF


    # Now we need to recover the tree by traversing the chart starting at the
    # root. This iterative implementation is faster than any of my attempts to
    # use helper functions and recursion

    # All fully binarized trees have the same number of nodes
    #cdef int num_tree_nodes = 2 * sentence_len - 1
    #cdef np.ndarray[int, ndim=1] included_i = np.empty(num_tree_nodes, dtype=np.int32)
    #cdef np.ndarray[int, ndim=1] included_j = np.empty(num_tree_nodes, dtype=np.int32)
    #cdef np.ndarray[int, ndim=1] included_label = np.empty(num_tree_nodes, dtype=np.int32)
    included_i=[]
    included_j=[]
    included_label=[]

    idx = 0
    stack_idx = 1
    # technically, the maximum stack depth is smaller than this
    #cdef np.ndarray[int, ndim=1] stack_i = np.empty(num_tree_nodes + 5, dtype=np.int32)
    #cdef np.ndarray[int, ndim=1] stack_j = np.empty(num_tree_nodes + 5, dtype=np.int32)
    stack_i=np.zeros(10*n_frame+4)
    stack_j=np.zeros(10*n_frame+4)
    stack_label=np.zeros((10*n_frame+4), dtype=np.int32)
    stack_i[1] = 0
    stack_j[1] = n_frame 
    stack_label[1] = 3 #root must in activity level(?)

    #cdef int i, j, k
    while stack_idx > 0:
        i = int(stack_i[stack_idx])
        j = int(stack_j[stack_idx])
        toplabel=int(stack_label[stack_idx])
        stack_idx -= 1
        #best_label_chart[i, j]!=0
        included_i.append(i)
        included_j.append(j)
        #print(i,j)
        if toplabel==0:
            topbot=0
        elif toplabel==1:
            topbot=np.argmax([NEG_INF,value_chart[i, j, 1],NEG_INF,NEG_INF,value_chart[i, j, 4]])
        elif toplabel==2:
            topbot=np.argmax([NEG_INF,NEG_INF,value_chart[i, j, 2],NEG_INF,NEG_INF,value_chart[i, j, 5],NEG_INF,value_chart[i, j, 7]])
        elif toplabel==3:
            topbot=np.argmax([NEG_INF,NEG_INF,NEG_INF,value_chart[i, j, 3],NEG_INF,NEG_INF,value_chart[i, j, 6],NEG_INF,value_chart[i, j, 8],value_chart[i, j, 9]])
        included_label.append(best_label_chart[i, j, topbot])
        idx += 1
        if i + 1 < j:
            k = split_idx_chart[i, j, label2top(topbot)]
            stack_idx += 1
            stack_i[stack_idx] = k
            stack_j[stack_idx] = j
            stack_label[stack_idx]=label2top(topbot)
            stack_idx += 1
            stack_i[stack_idx] = i
            stack_j[stack_idx] = k
            stack_label[stack_idx]=label2top(topbot)

    running_total = 0.0
    for idx in range(len(included_i)):
        running_total += label_scores_chart[included_i[idx], included_j[idx], included_label[idx]]

    #print(running_total)
    score = max(value_chart[0, n_frame ])
    #print(score)
    augment_amount = round(score - running_total)

    return score, np.array(included_i), np.array(included_j), np.array(included_label), augment_amount

