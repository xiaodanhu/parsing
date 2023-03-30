predicteds = flatten(predict_list)
groundtruths = flatten(gt_list)

n_samples=len(groundtruths)
for m in range(n_samples): #trees
    groundtruth=parser.evaluator.unpack(groundtruths[m])
    predicted=parser.evaluator.unpack(predicteds[m])
    gt_actions=groundtruth.keys()
    for action in gt_actions:
        parser.evaluator.gt_action_count[action]+=len(groundtruth[action])
    #gt_frames=[[int(j) for j in i if j !=0] for i in groundtruth[m][3].numpy().reshape(gt_shape[0]*gt_shape[1]*gt_shape[2],gt_shape[3]) if i[0]!=0]
    #assert len(gt_actions)==len(gt_frames)
    #n_gt=len(gt_actions)
    for action,duration in predicted.items():
        for frames in duration:
            TP_addition=np.zeros(parser.evaluator.intervals)
            parser.evaluator.pred_action_count[action]+=1
            if action in gt_actions:
                for gt_frames in groundtruth[action]:
            #for n in range(n_gt):
            #    if action==gt_actions[n]:
                    u_frames=set(gt_frames).union(set(frames))
                    i_frames=set(gt_frames).intersection(set(frames))
                    IoU=len(i_frames)/len(u_frames)
                    for x in range(parser.evaluator.intervals):
                        if IoU>=0.5:
                            TP_addition[x]=1
                            #parser.evaluator.Precision[action,x]+=1
            TP[action,:]=TP[action,:]+TP_addition