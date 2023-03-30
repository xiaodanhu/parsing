''' Usage

import prob_tree

pt=prob_tree.prob_tree()
pt.prior([3,4,4]) # prob of video with activity 3 4 4, list of activity(list of int) -> prob(float)
pt.activity2p(4,[41, 42, 41, 44, 42, 41, 43, 44, 43, 42]) # activity(int), list of phrase(list of int) -> prob(float)
pt.p2action(41,[77, 79, 79, 87, 88, 88])  # phrase(int),list of action(list of int) -> prob(float)
pt.action2frame(93,6)  # action(int),length(int) -> prob

'''

import json
class prob_tree(object):
    def __init__(self):
        with open("/data/xiaodan8/research/self-attentive-parser-v2/data/pt_prior2activity.json", "r") as file:
            self.pr2a=json.loads(file.read())
        with open("/data/xiaodan8/research/self-attentive-parser-v2/data/pt_activity2p.json", "r") as file:
            self.a2p=json.loads(file.read())
        with open("/data/xiaodan8/research/self-attentive-parser-v2/data/pt_p2action.json", "r") as file:
            self.p2a=json.loads(file.read())
        with open("/data/xiaodan8/research/self-attentive-parser-v2/data/pt_action2frames.json", "r") as file:
            self.a2f=json.loads(file.read())

    def prior(self,a):
        v=tuple(a)
        v=str(v)
        if v not in self.pr2a:
            return 0
        return self.pr2a[v]

    def activity2p(self,a,v):
        v=tuple(v)
        v=str(v)
        if str(a) not in self.a2p:
            return 0 
        if v not in self.a2p[str(a)]:
            return 0
        return self.a2p[str(a)][v]

    def p2action(self,v,a):
        a=tuple(a)
        a=str(a)
        if str(v) not in self.p2a:
            return 0 
        if a not in self.p2a[str(v)]:
            return 0
        return self.p2a[str(v)][a]
    
    def action2frame(self,v,a):
        if str(v) not in self.a2f:
            return 0
        if str(a) not in self.a2f[str(v)]:
            return 0
        return self.a2f[str(v)][str(a)]

