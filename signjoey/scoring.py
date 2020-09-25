import numpy
import pdb
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from utils.tools import torch_to_list, array_to_str
from nltk.translate.bleu_score import sentence_bleu

import sys
try:
    sys.path.append("cider")
    from pyciderevalcap.ciderD.ciderD import CiderD
    from pyciderevalcap.cider.cider import Cider
except:
    print('cider or coco-caption missing')

CiderD_scorer = None
Cider_scorer = None

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# TODO try knocking off EOS token
def calc_batch_bleu(hyp, ref, config):
    # Get sentence lists
    #pdb.set_trace()
    scores = []
    if config['batch_size'] == 1:
        Bleu_1 = sentence_bleu([ref[:-1]], hyp[:-1], weights=(1, 0, 0, 0))
        Bleu_2 = sentence_bleu([ref[:-1]], hyp[:-1], weights=(0, 1, 0, 0))
        Bleu_3 = sentence_bleu([ref[:-1]], hyp[:-1], weights=(0, 0, 1, 0))
        Bleu_4 = sentence_bleu([ref[:-1]], hyp[:-1], weights=(0, 0, 0, 1))
    else:
        Bleu_1 = 0
        Bleu_2 = 0
        Bleu_3 = 0
        Bleu_4 = 0
        #pdb.set_trace()
        for i in range(config['batch_size']):
            Bleu_1 += sentence_bleu([ref[i][:-1]], hyp[i][:-1], weights=(1, 0, 0, 0))
            Bleu_2 += sentence_bleu([ref[i][:-1]], hyp[i][:-1], weights=(0, 1, 0, 0))
            Bleu_3 += sentence_bleu([ref[i][:-1]], hyp[i][:-1], weights=(0, 0, 1, 0))
            Bleu_4 += sentence_bleu([ref[i][:-1]], hyp[i][:-1], weights=(0, 0, 0, 1))

    scores.append(100*Bleu_1/config['batch_size'])
    scores.append(100*Bleu_2/config['batch_size'])
    scores.append(100*Bleu_3/config['batch_size'])
    scores.append(100*Bleu_4/config['batch_size'])

    return scores

def init_scorer():
   # pdb.set_trace()
    global CiderD_scorer
    CiderD_scorer = CiderD(df='RWTH-words')
    global Cider_scorer
    Cider_scorer = Cider(df='RWTH-words')

# Gets the reward for a batch of sentences
def get_self_critical_reward(greedy_res, data_gts, gen_result, vocab_dict, config):
    # Prepare gts
    captions_reward = data_gts.cpu().numpy().astype('uint32')
    captions_arr = []
    for i in range(len(captions_reward)):
        captions_arr.append([captions_reward[i]])
    captions_indices = torch.arange(0, len(captions_arr))
    gts = [captions_arr[_] for _ in captions_indices.tolist()]
    # Define important measures
    batch_size = len(gts)
    gen_res_size = gen_result.shape[0]
    seq_per_img = gen_res_size // len(gts)

    gen_res_decode = torch_to_list(gen_result, vocab_dict, config)
    greedy_res_decode = torch_to_list(greedy_res, vocab_dict, config)
    gt_decode = torch_to_list(data_gts, vocab_dict, config)
    #pdb.set_trace()
    # Convert data_gts to list of numpys
    #test = gen_res_decode[0][1:]
    #separator = ' '
    #test_2 = separator.join(test)
    res = OrderedDict()
    for i in range(gen_res_size): # Put both captions into ordered dict
        res[i] = [(array_to_str(gen_res_decode[i][1:]))] # Trying [i][:] instead of [i][1:]
    for i in range(batch_size):
        res[gen_res_size + i] = [array_to_str(greedy_res_decode[i][1:])]
    
    #pdb.set_trace()
    gts = OrderedDict() # Put gts into similar format ordered dict
    for i in range(batch_size):
        gts[i] = [array_to_str(gt_decode[i])]# for j in range(len(data_gts[i]))]
    
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_res_size)}
    gts_.update({i+gen_res_size: gts[i] for i in range(batch_size)})

    #pdb.set_trace()

    avg_scores, cider_scores = CiderD_scorer.compute_score(gts_, res_)

    scores = cider_scores
    #pdb.set_trace()
    scores = scores[:gen_res_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_res_size)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

# Calculates the CIDEr score for a batch of sentences
def calc_batch_cider(hyp, ref, vocab_dict, config):
    batch_size = config['batch_size']
    gen_res_size = hyp.shape[0]
    seq_per_img = gen_res_size // len(ref)

    hyp_decode = torch_to_list(hyp, vocab_dict, config)
    gt_decode = torch_to_list(ref, vocab_dict, config)

    res = OrderedDict()
    for i in range(gen_res_size): # Put both captions into ordered dict
        res[i] = [array_to_str(hyp_decode[i][1:])]
    
    gts = OrderedDict() # Put gts into similar format ordered dict
    for i in range(batch_size):
        gts[i] = [array_to_str(gt_decode[i])]# for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_res_size)}

    #pdb.set_trace()
    avg_scores, cider_scores = CiderD_scorer.compute_score(gts_, res_)

    return avg_scores # used to be *100

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward): # (sample_logprobs, gen_result.data, reward)
        #pdb.set_trace() ### DEBUG: this is from self.rl_crit, calculates final gradient expression
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq>0).float() # Purpose: have '1s' where there are actual words and '0s' where there are no words in sequence
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output