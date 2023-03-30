import numpy as np

class mAP_Evaluator(object):
    def __init__(self,vocabulary,st_IoU=0.3,ed_IoU=0.95,intervals=14):
        n_action=len(vocabulary)-3
        self.vocabulary=vocabulary
        self.IoUs=np.linspace(st_IoU,ed_IoU,intervals)
        self.TP=np.zeros((n_action,intervals))
        self.Precision=np.zeros((n_action,intervals))
        self.Recall=np.zeros((n_action,intervals))
        self.intervals=intervals
        self.n_action=n_action
        self.AP=np.zeros(n_action)
        self.gt_action_count=np.zeros(n_action)
        self.pred_action_count=np.zeros(n_action)
        self.mAP=0
        #self.masks=np.zeros(n_action)

    def unpack(self,data):
        vocabulary=self.vocabulary
        non_act=[vocabulary['BG_Action'],vocabulary['<START>'],vocabulary['<STOP>']]
        actions={}
        old=-1
        duration=[]
        for i in range(len(data)):
            if data[i]!=old:
                if len(duration)>0:
                    if old in actions:
                        actions[old].append(duration)
                    elif (old not in actions) and (old not in non_act):
                        actions[old]=[duration]
                duration=[]
                old=data[i]
            
            duration.append(i)
        if len(duration)>0:
            if old in actions:
                actions[old].append(duration)
            elif (old not in actions) and (old not in non_act):
                actions[old]=[duration]
        return actions

    def evaluate(self,groundtruths,predicteds):
        assert len(groundtruths)==len(predicteds)
        n_samples=len(groundtruths)
        #gt_shape=groundtruth[0][3].shape
        #pred_action_count=np.zeros(self.n_action)
        #gt_action_count=np.zeros(self.n_action)
        for m in range(n_samples): #trees
            groundtruth=self.unpack(groundtruths[m])
            predicted=self.unpack(predicteds[m])
            gt_actions=groundtruth.keys()
            for action in gt_actions:
                self.gt_action_count[action]+=len(groundtruth[action])
            #gt_frames=[[int(j) for j in i if j !=0] for i in groundtruth[m][3].numpy().reshape(gt_shape[0]*gt_shape[1]*gt_shape[2],gt_shape[3]) if i[0]!=0]
            #assert len(gt_actions)==len(gt_frames)
            #n_gt=len(gt_actions)
            for action,duration in predicted.items():
                for frames in duration:
                    TP_addition=np.zeros(self.intervals)
                    self.pred_action_count[action]+=1
                    if action in gt_actions:
                        for gt_frames in groundtruth[action]:
                    #for n in range(n_gt):
                    #    if action==gt_actions[n]:
                            u_frames=set(gt_frames).union(set(frames))
                            i_frames=set(gt_frames).intersection(set(frames))
                            IoU=len(i_frames)/len(u_frames)
                            for x in range(self.intervals):
                                if IoU>=self.IoUs[x]:
                                    TP_addition[x]=1
                                    #self.Precision[action,x]+=1
                    self.TP[action,:]=self.TP[action,:]+TP_addition
        #self.Recall=self.Precision/gt_action_count
        #self.Precision=self.Precision/pred_action_count
        for i in range(self.n_action):
            if self.gt_action_count[i]==0:
                self.Recall[i,:]=0
            else:
                self.Recall[i,:]=self.TP[i,:]/self.gt_action_count[i]
            if self.pred_action_count[i]==0:
                self.Precision[i,:]=0
            else:
                self.Precision[i,:]=self.TP[i,:]/self.pred_action_count[i]
            ###
            mprec = np.hstack([[0], self.Precision[i], [0]])
            mrec = np.hstack([[0], self.Recall[i], [1]])
            for j in range(len(mprec) - 1)[::-1]:
                mprec[j] = max(mprec[j], mprec[j + 1])
            idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
            self.AP[i] = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]) 
            ###       
        if len([i for i in self.pred_action_count if i !=0]) == 0:
            self.mAP=0
        else:
            self.mAP=sum([self.AP[i] for i in range(self.n_action) if self.pred_action_count[i]!=0])/len([i for i in self.pred_action_count if i !=0])
    def AP_at(self,interval):
        assert(interval<self.intervals)
        return self.Precision[:,interval].mean()