import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import argparse
import itertools
import json
import os.path
import time

import torch
import torch.optim.lr_scheduler

import numpy as np

import evaluate
import eval
import glob
from sklearn.metrics import average_precision_score

# import treesvideoBG_FG as treesvideo_FG
# import dataloaderBG_FG as dataloader_FG
# import treesvideoBG_FA as treesvideo_FA
# import dataloaderBG_FA as dataloader_FA

import vocabulary
import nkutil
import parse_nk
import wandb

tokens = parse_nk


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap

def flatten(x):
    return [item for sublist in x for item in sublist]

def evaluate_action_recog(pre, gt, vids, vocab_size):
    pre, gt, vids = flatten(pre), flatten(gt), flatten(vids)
    v_set = set(vids)
    merged_pre_list = []
    merged_gt_list = []
    for v in v_set:
        index = [i for i, x in enumerate(vids) if x == v]
        merged_pre = torch.zeros(vocab_size)
        merged_gt = torch.zeros(vocab_size)
        merged_pre[list(set([pre[i] for i in index]))] = 1
        merged_gt[list(set([gt[i] for i in index]))] = 1
        merged_pre_list.append(merged_pre)
        merged_gt_list.append(merged_gt)

    return get_map(np.vstack(merged_pre_list)[:,:-3], np.vstack(merged_gt_list)[:,:-3])


def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return nkutil.HParams(
        max_len_train=300, # no length limit
        max_len_dev=300, # no length limit

        sentence_max_len=300,

        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=10, # 3 establishes a termination criterion

        partitioned=True,
        num_layers_position_only=0,

        d_model= 1024,
        num_heads= 8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_tag_hidden=250,
        tag_loss_scale=5.0,

        attention_dropout=0.2,
        embedding_dropout=0.0,
        relu_dropout=0.1,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_chars_lstm=False,
        use_elmo=False,
        use_bert=False,
        use_bert_only=False,
        predict_tags=False,

        d_char_emb=32, # A larger value may be better for use_chars_lstm

        tag_emb_dropout= 0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-base-uncased",
        bert_do_lower_case=True,
        bert_transliterate="",
        )

def run_train(args, hparams):

    if args.dataset == "FineGym":
        import treesvideoBG_FG as treesvideo
        import dataloaderBG_FG as dataloader
    else:
        import treesvideoBG_FA as treesvideo
        import dataloaderBG_FA as dataloader

    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", 1872393927)
    torch.manual_seed(1872393927)

    hparams.set_from_args(args)
    hparams.learning_rate = args.learning_rate
    hparams.num_layers = args.num_layers
    hparams.decode_method = args.decode_method
    hparams.is_inference = args.is_inference
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    # print("Hyperparameters:")
    # hparams.print()

    print("Loading dataset")
    path ="/data/xiaodan8/research/dataset/" + args.dataset
    loaders = dataloader.construct_dataloaders(path, path+'/I3D_features', 32, 0, 44739242,3,10,10,20)

    print("Loading training trees from {}...".format(args.train_path))
    if args.task == 'video':
        train_treebank = treesvideo.load_trees_wBG(mode='train',path='/data/xiaodan8/research/dataset/' + args.dataset, loaders=loaders)
    elif args.task == 'text':
        train_treebank = trees.load_trees(args.train_path)

    if hparams.max_len_train > 0:
        train_treebank = [(tree, vid) for tree,vid in train_treebank if len(list(tree.leaves())) <= hparams.max_len_train]
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    if args.task == 'video':
        dev_treebank = treesvideo.load_trees_wBG(mode='test',path='/data/xiaodan8/research/dataset/' + args.dataset, loaders=loaders)
    elif args.task == 'text':
        dev_treebank = trees.load_trees(args.dev_path)

    if hparams.max_len_dev > 0:
        dev_treebank = [(tree, vid) for tree,vid in dev_treebank if len(list(tree.leaves())) <= hparams.max_len_dev]
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [(tree.convert(), vid) for tree,vid in train_treebank]
    dev_parse = [(tree.convert(), vid) for tree,vid in dev_treebank]

    print("Constructing vocabularies...")
    id2actionlabel=loaders['train'].dataset.id2actionlabel
    id2phraselabel=loaders['train'].dataset.id2phraselabel
    id2activitylabel=loaders['train'].dataset.id2activitylabel
    action2phr = {id2actionlabel[k]:id2phraselabel[v] for k,v in loaders['train'].dataset.action2phr.items()}
    action2phr.update({'BG_Action': 'BG_Phrase'}) # add 'BG_Action' to 'BG_Phrase'

    activity_vocab=vocabulary.Vocabulary()
    for i in id2activitylabel:
        activity_vocab.index(id2activitylabel[i])
    activity_vocab.index('BG_Activity')
    activity_vocab.index('<START>')
    activity_vocab.index('<STOP>')
    activity_vocab.freeze()

    phrase_vocab=vocabulary.Vocabulary()
    for i in id2phraselabel:
        phrase_vocab.index(id2phraselabel[i])
    phrase_vocab.index('BG_Phrase')
    phrase_vocab.index('<START>')
    phrase_vocab.index('<STOP>')
    phrase_vocab.freeze()

    action_vocab=vocabulary.Vocabulary()
    for i in id2actionlabel:
        action_vocab.index(id2actionlabel[i])
    action_vocab.index('BG_Action')
    action_vocab.index('<START>')
    action_vocab.index('<STOP>')
    action_vocab.freeze()


    with wandb.init():
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # args.batch_size = config.batch_size

        hparams.weight_action = config.weight_action
        hparams.weight_phrase = config.weight_phrase
        hparams.weight_activity = config.weight_activity
        # hparams.learning_rate = config.learning_rate
        # hparams.num_layers = config.num_layers
        # hparams.d_model= config.d_model # 1024,
        # hparams.num_heads= config.num_heads # 8,
        # hparams.tag_emb_dropout= config.tag_emb_dropout # 0.2,

        print("Initializing model...")

        load_path = None
        if load_path is not None:
            print(f"Loading parameters from {load_path}")
            info = torch_load(load_path)
            parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
        else:
            parser = parse_nk.NKChartParser(
                activity_vocab,
                phrase_vocab,
                action_vocab,
                id2phraselabel,
                hparams,
            )

        print("Initializing optimizer...")
        trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
        trainer = torch.optim.Adam(trainable_parameters, lr=0.01, betas=(0.9, 0.98), eps=1e-9)
        if load_path is not None:
            trainer.load_state_dict(info['trainer'])

        def set_lr(new_lr):
            for param_group in trainer.param_groups:
                param_group['lr'] = new_lr

        assert hparams.step_decay, "Only step_decay schedule is supported"

        warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer, 'max',
            factor=hparams.step_decay_factor,
            patience=hparams.step_decay_patience,
            verbose=True,
        )
        def schedule_lr(iteration):
            iteration = iteration + 1
            if iteration <= hparams.learning_rate_warmup_steps:
                set_lr(iteration * warmup_coeff)

        clippable_parameters = trainable_parameters
        grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 else hparams.clip_grad_norm

        print("Training...")
        total_processed = 0
        current_processed = 0
        check_every = len(train_parse) / args.checks_per_epoch
        best_dev_fscore = -np.inf
        best_dev_model_path = None
        best_dev_processed = 0

        start_time = time.time()


        def check_dev():
            nonlocal best_dev_fscore
            nonlocal best_dev_model_path
            nonlocal best_dev_processed

            dev_start_time = time.time()

            parser.evaluator=eval.mAP_Evaluator(action_vocab.indices)
            predict_list = []
            gt_list = []
            vid_list = []
            for dev_start_index in range(0, len(dev_treebank), args.eval_batch_size):
                subbatch_trees = dev_treebank[dev_start_index:dev_start_index+args.eval_batch_size]
                subbatch_trees = [(act, vid) for tree, vid in subbatch_trees for act in tree.children]
                subbatch_sentences = [[(leaf.tag, action2phr[leaf.tag], tree.label, leaf.I3D) for leaf in tree.leaves()] for tree, _ in subbatch_trees]
                acc, loss, mAP, predict, gt = parser.parse_batch(subbatch_sentences)
                predict_list.append(predict)
                gt_list.append(gt)
                vid_list.append([vid for tree, vid in subbatch_trees for _ in tree.leaves()])

            dev_recog_mAP = evaluate_action_recog(predict_list, gt_list, vid_list, action_vocab.size())
            dev_local_mAP = parser.evaluator.mAP


            parser.evaluator=eval.mAP_Evaluator(action_vocab.indices)
            predict_list = []
            gt_list = []
            vid_list = []
            for train_start_index in range(0, len(train_treebank), args.eval_batch_size):
                subbatch_trees = train_treebank[train_start_index:train_start_index+args.eval_batch_size]
                subbatch_trees = [(act, vid) for tree, vid in subbatch_trees for act in tree.children]
                subbatch_sentences = [[(leaf.tag, action2phr[leaf.tag], tree.label, leaf.I3D) for leaf in tree.leaves()] for tree, _ in subbatch_trees]
                acc, loss, mAP, predict, gt = parser.parse_batch(subbatch_sentences)
                predict_list.append(predict)
                gt_list.append(gt)
                vid_list.append([vid for tree, vid in subbatch_trees for _ in tree.leaves()])

            train_recog_mAP = evaluate_action_recog(predict_list, gt_list, vid_list, action_vocab.size())
            train_local_mAP = parser.evaluator.mAP

            print(
                "dev-Recognition mAP {:.4f} "
                "dev-Localization mAP {:.4f} "
                "train-Recognition mAP {:.4f} "
                "train-Localization mAP {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    dev_recog_mAP,
                    dev_local_mAP,
                    train_recog_mAP,
                    train_local_mAP,
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time),
                )
            )

            if dev_local_mAP > best_dev_fscore:
                if best_dev_model_path is not None:
                    extensions = [".pt"]
                    for ext in extensions:
                        path = best_dev_model_path + ext
                        if os.path.exists(path):
                            print("Removing previous model file {}...".format(path))
                            os.remove(path)

                best_dev_fscore = dev_local_mAP
                best_dev_model_path = "{}_dev={:.2f}".format(
                    args.model_path_base, dev_local_mAP)
                best_dev_processed = total_processed
                print("Saving new best model to {}...".format(best_dev_model_path))
                torch.save({
                    'spec': parser.spec,
                    'state_dict': parser.state_dict(),
                    'trainer' : trainer.state_dict(),
                    }, best_dev_model_path + ".pt")
            
            return dev_local_mAP

        dev_mAP = 0
        for epoch in itertools.count(start=1):
            if args.epochs is not None and epoch > args.epochs:
                break

            # import eval
            parser.evaluator=eval.mAP_Evaluator(action_vocab.indices)

            np.random.shuffle(train_parse)
            epoch_start_time = time.time()

            for start_index in range(0, len(train_parse), args.batch_size):
                trainer.zero_grad()
                schedule_lr(total_processed // args.batch_size)

                batch_loss_value = 0.0
                train_accuracy_action, train_accuracy_phrase, train_accuracy_activity = [], [], []
                batch_trees = train_parse[start_index:start_index + args.batch_size]
                batch_trees = [(act, vid) for tree, vid in batch_trees for act in tree.children]
                # batch_sentences = [[[(leaf.tag, action2phr[leaf.tag], act.label[0], leaf.I3D) for leaf in act.leaves()] for act in tree.children] for tree in batch_trees]
                # batch_sentences = [item for sublist in batch_sentences for item in sublist]
                batch_sentences = [[(leaf.tag, action2phr[leaf.tag], tree.label[0], leaf.I3D) for leaf in tree.leaves()] for tree, _ in batch_trees]
                batch_num_tokens = sum(len(sentence) for sentence in batch_sentences)
                # print("Batch Number tokens -------------> ", batch_num_tokens)

                for subbatch_sentences, subbatch_trees in parser.split_batch(batch_sentences, batch_trees, args.subbatch_max_tokens):
                    train_accuracy, loss, mAP, pred, gt = parser.parse_batch(subbatch_sentences, subbatch_trees)
                    loss = loss / len(batch_trees)
                    loss_value = float(loss.data.cpu().numpy())
                    # print(loss_value)
                    batch_loss_value += loss_value
                    if loss_value > 0:
                        loss.backward()
                    del loss
                    total_processed += len(subbatch_trees)
                    current_processed += len(subbatch_trees)
                    train_accuracy_action.append(train_accuracy[0])
                    train_accuracy_phrase.append(train_accuracy[1])
                    train_accuracy_activity.append(train_accuracy[2])

                grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, grad_clip_threshold)

                trainer.step()

                print(
                    "epoch {:,} "
                    "batch {:,}/{:,} "
                    "processed {:,} "
                    "batch-loss {:.4f} "
                    "mAP {:.4f} "
                    "action acc {:.4f} "
                    "phrase acc {:.4f} "
                    "activity acc {:.4f} "
                    "grad-norm {:.4f} "
                    "epoch-elapsed {} "
                    "total-elapsed {}".format(
                        epoch,
                        start_index // args.batch_size + 1,
                        int(np.ceil(len(train_parse) / args.batch_size)),
                        total_processed,
                        batch_loss_value,
                        mAP,
                        torch.mean(torch.tensor(train_accuracy_action)),
                        torch.mean(torch.tensor(train_accuracy_phrase)),
                        torch.mean(torch.tensor(train_accuracy_activity)),
                        grad_norm,
                        format_elapsed(epoch_start_time),
                        format_elapsed(start_time),
                    )
                )

                
                if current_processed >= check_every:
                    current_processed -= check_every
                    dev_mAP = check_dev()

                wandb.log({"epoch": epoch, "batch": start_index // args.batch_size + 1, "mAP": dev_mAP})

            # adjust learning rate at the end of an epoch
            if (total_processed // args.batch_size + 1) > hparams.learning_rate_warmup_steps:
                scheduler.step(best_dev_fscore)
                if (total_processed - best_dev_processed) > ((hparams.step_decay_patience + 1) * hparams.max_consecutive_decays * len(train_parse)):
                    print("Terminating due to lack of improvement in dev fscore.")
                    break

        # 🐝 Close your wandb run 
        wandb.finish()


def run_test(args, hparams):

    if args.dataset == "FineGym":
        import treesvideoBG_FG as treesvideo
        import dataloaderBG_FG as dataloader
    else:
        import treesvideoBG_FA as treesvideo
        import dataloaderBG_FA as dataloader

    hparams.set_from_args(args)
    hparams.num_layers = args.num_layers
    hparams.decode_method = args.decode_method
    hparams.is_inference = args.is_inference
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    print("Loading dataset")
    path ="/data/xiaodan8/research/dataset/" + args.dataset
    loaders = dataloader.construct_dataloaders(path, path+'/I3D_features', 32, 0, 44739242,3,10,10,20)

    print("Loading development trees")
    test_treebank = treesvideo.load_trees_wBG(mode='test',path='/data/xiaodan8/research/dataset/' + args.dataset, loaders=loaders)

    if hparams.max_len_dev > 0:
        test_treebank = [(tree, vid) for tree,vid in test_treebank if len(list(tree.leaves())) <= hparams.max_len_dev]
    print("Loaded {:,} development examples.".format(len(test_treebank)))

    print("Constructing vocabularies...")
    id2actionlabel=loaders['train'].dataset.id2actionlabel
    id2phraselabel=loaders['train'].dataset.id2phraselabel
    id2activitylabel=loaders['train'].dataset.id2activitylabel
    action2phr = {id2actionlabel[k]:id2phraselabel[v] for k,v in loaders['train'].dataset.action2phr.items()}
    action2phr.update({'BG_Action': 'BG_Phrase'}) # add 'BG_Action' to 'BG_Phrase'

    activity_vocab=vocabulary.Vocabulary()
    for i in id2activitylabel:
        activity_vocab.index(id2activitylabel[i])
    activity_vocab.index('BG_Activity')
    activity_vocab.index('<START>')
    activity_vocab.index('<STOP>')
    activity_vocab.freeze()

    phrase_vocab=vocabulary.Vocabulary()
    for i in id2phraselabel:
        phrase_vocab.index(id2phraselabel[i])
    phrase_vocab.index('BG_Phrase')
    phrase_vocab.index('<START>')
    phrase_vocab.index('<STOP>')
    phrase_vocab.freeze()

    action_vocab=vocabulary.Vocabulary()
    for i in id2actionlabel:
        action_vocab.index(id2actionlabel[i])
    action_vocab.index('BG_Action')
    action_vocab.index('<START>')
    action_vocab.index('<STOP>')
    action_vocab.freeze()

    print("Loading model from {}...".format(args.model_path_base))
    info = torch_load(sorted(glob.glob(args.model_path_base+'*.pt'))[-1])
    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
    parser.evaluator=eval.mAP_Evaluator(action_vocab.indices)
    
    print("Parsing test sentences...")
    start_time = time.time()
    test_mAP=0.0
    predict_list = []
    gt_list = []
    vid_list = []
    for start_index in range(0, len(test_treebank), args.eval_batch_size):
        subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
        subbatch_trees = [(act, vid) for tree, vid in subbatch_trees for act in tree.children]
        subbatch_sentences = [[(leaf.tag, action2phr[leaf.tag], tree.label, leaf.I3D) for leaf in tree.leaves()] for tree, _ in subbatch_trees]
        acc, loss, mAP, predict, gt = parser.parse_batch(subbatch_sentences)
        predict_list.append(predict)
        gt_list.append(gt)
        vid_list.append([vid for tree, vid in subbatch_trees for _ in tree.leaves()])

    print(
        "test-action recognition mAP {:.4f} "
        "test-action localization mAP {:.4f} ".format(
            evaluate_action_recog(predict_list, gt_list, vid_list, action_vocab.size()),
            parser.evaluator.mAP
        )
    )


def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", default=" ")
    subparser.add_argument("--evalb-dir", default="evalb/")
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=8, help='250 by default')
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=100)
    subparser.add_argument("--epochs", type=int, default=100)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--task", type=str, default="video")
    subparser.add_argument("--exp-name", type=str, default="Nov10")
    subparser.add_argument("--learning-rate", default=0.0008, type=float)
    subparser.add_argument("--num-layers", default=8, type=int, help='8 by default')
    subparser.add_argument("--output-dir", default="tmp/")
    subparser.add_argument("--dataset", default=" ", help='FineGym or FineAction')
    subparser.add_argument("--decode-method", default="transformer", help='transformer or linear')
    subparser.add_argument("--is-inference", default=False)
    subparser.set_defaults(use_words=True)
    subparser.set_defaults(use_chars_lstm=True)
    subparser.set_defaults(d_char_emb=64)

    subparser.set_defaults(dataset="FineGym") # FineGym or FineAction
    subparser.set_defaults(model_path_base="models/finegym_transformer") # finegym_linear or finegym_transformer

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", default=" ")
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--test-path-raw", type=str)
    subparser.add_argument("--eval-batch-size", type=int, default=100)
    subparser.add_argument("--num-layers", default=8, type=int, help='8 by default')
    subparser.add_argument("--exp-name", type=str, default="Mar5")
    subparser.add_argument("--output-dir", default="tmp/")
    subparser.add_argument("--dataset", default=" ")
    subparser.add_argument("--decode-method", default="transformer", help='transformer or linear')
    subparser.add_argument("--is-inference", default=True)

    subparser.set_defaults(dataset="FineGym") # FineGym or FineAction
    subparser.set_defaults(model_path_base="models/finegym_transformer") # finegym_linear or finegym_transformer

    args = parser.parse_args(['train'])
    run_train(args, hparams)
    
    # args = parser.parse_args(['test'])
    # run_test(args, hparams)


if __name__ == "__main__":


    wandb.login()

    metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }

    sweep_config = {
        'method': 'random',
        'metric': metric
        }

    parameters_dict = {
        'optimizer': {
            'values': ['adam']
            },
        # 'tag_emb_dropout': {
        #     'values': [0.2, 0.3, 0.4]
        #     },
        }

    parameters_dict.update({
        # 'learning_rate': {
        #     # a flat distribution between 0 and 0.1
        #     'distribution': 'uniform',
        #     'min': 0,
        #     'max': 1
        # },
        'weight_action': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 1,
            'max': 10
        },
        'weight_phrase': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 1,
            'max': 10
        },
        'weight_activity': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 1,
            'max': 10
        },
        # 'batch_size': {
        #     'distribution': 'q_log_uniform_values',
        #     'q': 8,
        #     'min': 8,
        #     'max': 8,
        # },
        # 'num_layers': {'values': [2, 4, 8, 16]},
        # 'd_model': {
        #     'distribution': 'q_log_uniform_values',
        #     'q': 8,
        #     'min': 512,
        #     'max': 2048,
        # },
        # 'num_heads': {'values': [4, 8]},
        })


    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="parsing")
    wandb.agent(sweep_id, main, count=100)