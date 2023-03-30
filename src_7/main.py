import numpy as np
import json
import prob_tree

def extract(string):
    return list(map(int, filter(None, string[1:-1].split(","))))

def id2index(id):
    return {i:idx for idx, i in enumerate(id)}

pt = prob_tree.prob_tree()

sub_act_probs = np.random.random((200, 99))

prob_tree = pt.p2a
id2idx = id2index(prob_tree.keys())
n_activity = len(prob_tree)
acts = prob_tree.keys()
start = 0
act_probs = [0] * n_activity
from collections import defaultdict
sub_act_out = defaultdict(list)
candidate_length_out = defaultdict(list)

# sub_acts = [[2,3],[3,4,5]]
# n_levels = len(sub_acts)
# candidate_lengths = []

def recurse(level, res):
    global candidate_lengths

    if level == len(sub_acts_lengths):
        candidate_lengths.append(res[:])
        return
    
    for i in sub_acts_lengths[level]:
        res.append(i)
        recurse(level + 1, res)
        res.pop()

# result = []
# recurse(0, [])
# print(result)

while start < len(sub_act_probs):
    for act_id, (act, rules_probs) in enumerate(prob_tree.items()):
        probs = []
        sub_acts_list = []
        candidate_lengths_list = []
        for rule, rule_prob in rules_probs.items():
            sub_acts = extract(rule)
            sub_acts_lengths = [list(map(int, pt.a2f[str(i)].keys())) for i in sub_acts]
            candidate_lengths = []
            recurse(0, [])
            candidate_sub_act_probs = [np.prod([np.prod(sub_act_probs[start : start + l, sub_acts[idx]]) for idx, l in enumerate(lengths)]) for lengths in candidate_lengths]
            sub_act_prob = [rule_prob * candidate_sub_act_probs[i] *  np.prod([pt.action2frame(sub_acts[idx],l) for idx, l in enumerate(lengths)]) for i, lengths in enumerate(candidate_lengths)]
            probs.append(sub_act_prob)
            sub_acts_list.append(sub_acts)
            candidate_lengths_list.append(candidate_lengths)
        max_value, max_index = max((x, (i, j)) for i, row in enumerate(probs) for j, x in enumerate(row))
        sub_act_out[id2idx[act]] = sub_acts_list[max_index[0]]
        candidate_length_out[id2idx[act]] = candidate_lengths_list[max_index[0]][max_index[1]]
        act_probs[id2idx[act]] = max_value
    max_act_id = np.argmax(act_probs)
    max_sub_act = sub_act_out[max_act_id]
    max_length = candidate_length_out[max_act_id]
    start += np.sum(max_length)

    
# while start < len(sub_act_probs):
#     for act_id, (act, rules_probs) in enumerate(prob_tree.items()):
#         probs = []
#         sub_acts_list = []
#         for rule, rule_prob in rules_probs.items():
#             sub_acts = extract(rule)
#             sub_act_prob = np.prod([sub_act_probs[start + i][sub_act] for i, sub_act in enumerate(sub_acts)])
#             prob = rule_prob * sub_act_prob
#             probs.append(prob)
#             sub_acts_list.append(sub_acts)
#         sub_acts = sub_acts_list[np.argmax(probs)]
#         sub_act_out.extend(sub_acts)
#         act_probs[int(act_id)] = np.max(probs)
#     start += len(sub_acts)