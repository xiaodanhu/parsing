import math
import os.path
import re
import subprocess
import tempfile
from PYEVALB import scorer
import treesvideo

class FScore(object):
    def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match
        self.tagging_accuracy = tagging_accuracy

    def __str__(self):
        if self.tagging_accuracy < 100:
            return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f}, TaggingAccuracy={:.2f})".format(
                self.recall, self.precision, self.fscore, self.complete_match, self.tagging_accuracy)
        else:
            return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f})".format(
                self.recall, self.precision, self.fscore, self.complete_match)

def evalb(evalb_dir, output_dir, gold_trees, predicted_trees, ref_gold_path=None):
    assert os.path.exists(evalb_dir)
    evalb_program_path = "evalb" #os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = "evalb_spmrl" #os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)

    if os.path.exists(evalb_program_path):
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    # for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
    #     assert isinstance(gold_tree, treesvideo.TreebankNode)
    #     assert isinstance(predicted_tree, treesvideo.TreebankNode)
    #     gold_leaves = list(gold_tree.leaves())
    #     predicted_leaves = list(predicted_tree.leaves())
        # assert len(gold_leaves) == len(predicted_leaves)
        # assert all(
        #     gold_leaf.word == predicted_leaf.word
        #     for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gold_path = os.path.join(output_dir, "gold.txt")
    predicted_path = os.path.join(output_dir, "predicted.txt")
    output_path = os.path.join(output_dir, "output.txt")
    # gold_path = os.path.join('tmp', "gold.txt")
    # predicted_path = os.path.join('tmp', "predicted.txt")
    # output_path = os.path.join('tmp', "output.txt")

    with open(gold_path, "w") as outfile:
        if ref_gold_path is None:
            for tree in gold_trees:
                outfile.write("{}\n".format(tree.linearize()))
        else:
            # For the SPMRL dataset our data loader performs some modifications
            # (like stripping morphological features), so we compare to the
            # raw gold file to be certain that we haven't spoiled the evaluation
            # in some way.
            with open(ref_gold_path) as goldfile:
                outfile.write(goldfile.read())

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    # command = "{} -p {} {} {} > {}".format(
    #     evalb_program_path,
    #     evalb_param_path,
    #     gold_path,
    #     predicted_path,
    #     output_path,
    # )
    # subprocess.run(command, shell=True)

    s = scorer.Scorer()
    try:
        s.evalb(gold_path, predicted_path, output_path)
    except:
        print(predicted_path)
    # result = s.score_trees(gold_trees, predicted_trees)
    # print('Recall =' + str(result.recall))
    # print('Precision =' + str(result.prec))

    fscore = FScore(math.nan, math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall:\t", line)
            if match:
                fscore.recall = float(line[match.regs[0][1]:])
            match = re.match(r"Bracketing Precision:\t", line)
            if match:
                fscore.precision = float(line[match.regs[0][1]:])
            match = re.match(r"Bracketing FMeasure:\t", line)
            if match:
                fscore.fscore = float(line[match.regs[0][1]:])
            match = re.match(r"Complete match:\t", line)
            if match:
                fscore.complete_match = float(line[match.regs[0][1]:])
            match = re.match(r"Tagging accuracy:\t", line)
            if match:
                fscore.tagging_accuracy = float(line[match.regs[0][1]:])
                break

    success = (
        not math.isnan(fscore.fscore) or
        fscore.recall == 0.0 or
        fscore.precision == 0.0)

    # if success:
    #     temp_dir.cleanup()
    # else:
    #     print("Error reading EVALB results.")
    #     print("Gold path: {}".format(gold_path))
    #     print("Predicted path: {}".format(predicted_path))
    #     print("Output path: {}".format(output_path))

    return fscore
