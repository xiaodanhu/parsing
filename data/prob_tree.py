import json
class prob_tree(object):
    def __init__(self):
        with open("pt_activity2p.json", "r") as file:
            self.a2p=json.loads(file.read())
        with open("pt_p2action.json", "r") as file:
            self.p2a=json.loads(file.read())

    def activity2p(self,a,v):
        v=tuple(v)
        v=str(v)
        return self.a2p[str(a)][v]

    def p2action(self,v,a):
        a=tuple(a)
        a=str(a)
        return self.p2a[str(v)][a]
    