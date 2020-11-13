import numpy
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from signjoey.tools import torch_to_list, array_to_str, list_to_string
from nltk.translate.bleu_score import sentence_bleu
from signjoey.metrics import wer_list, bleu
import wandb

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
def get_self_critical_reward(greedy_res, data_gts, gen_result, factor):
    # Prepare gts
    #pdb.set_trace()
    #captions_reward = data_gts.cpu().numpy().astype('uint32')
    #captions_arr = []
    #for i in range(len(captions_reward)):
    #    captions_arr.append([captions_reward[i]])
    #captions_indices = torch.arange(0, len(captions_arr))
    #gts = [captions_arr[_] for _ in captions_indices.tolist()]
    # Define important measures
    batch_size = len(data_gts)
    gen_res_size = len(gen_result)
    seq_per_img = gen_res_size // batch_size

    #gen_res_decode = torch_to_list(gen_result, vocab_dict, config)
    #greedy_res_decode = torch_to_list(greedy_res, vocab_dict, config)
    #gt_decode = torch_to_list(data_gts, vocab_dict, config)
    #pdb.set_trace()
    res = OrderedDict()
    for i in range(gen_res_size): # Put both captions into ordered dict
        res[i] = [(array_to_str(gen_result[i][1:]))] # Trying [i][:] instead of [i][1:]
    for i in range(batch_size):
        res[gen_res_size + i] = [array_to_str(greedy_res[i][1:])]
    
    #pdb.set_trace()
    gts = OrderedDict() # Put gts into similar format ordered dict
    for i in range(batch_size):
        gts[i] = [array_to_str(data_gts[i])]# for j in range(len(data_gts[i]))]
    
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_res_size)}
    gts_.update({i+gen_res_size: gts[i] for i in range(batch_size)})

    #pdb.set_trace()

    avg_score, cider_scores = CiderD_scorer.compute_score(gts_, res_)

    scores = cider_scores
    wandb.log({'train/avg_CIDEr_score': avg_score})
    scores = scores[:gen_res_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_res_size)
    #rewards = np.repeat(scores[:, np.newaxis], max(map(len, gen_result)), 1)
    rewards = np.repeat(scores[:, np.newaxis], factor, 1)
    ##########################################
    #pdb.set_trace()
    #if rewards.shape[1] != max(map(len, gen_result)):
    #    pdb.set_trace()

    return rewards

def get_self_critical_reward_bleu(greedy_res, data_gts, gen_result, factor):
    # Prepare gts
    #pdb.set_trace()
    #captions_reward = data_gts.cpu().numpy().astype('uint32')
    #captions_arr = []
    #for i in range(len(captions_reward)):
    #    captions_arr.append([captions_reward[i]])
    #captions_indices = torch.arange(0, len(captions_arr))
    #gts = [captions_arr[_] for _ in captions_indices.tolist()]
    # Define important measures
    batch_size = len(data_gts)
    gen_res_size = len(gen_result)
    seq_per_img = gen_res_size // batch_size

    #gen_res_decode = torch_to_list(gen_result, vocab_dict, config)
    #greedy_res_decode = torch_to_list(greedy_res, vocab_dict, config)
    #gt_decode = torch_to_list(data_gts, vocab_dict, config)
    #pdb.set_trace()
    # res = OrderedDict()
    # for i in range(gen_res_size): # Put both captions into ordered dict
    #     res[i] = [(array_to_str(gen_result[i][1:]))] # Trying [i][:] instead of [i][1:]
    # for i in range(batch_size):
    #     res[gen_res_size + i] = [array_to_str(greedy_res[i][1:])]

    # #pdb.set_trace()
    # gts = OrderedDict() # Put gts into similar format ordered dict
    # for i in range(batch_size):
    #     gts[i] = [array_to_str(data_gts[i])]# for j in range(len(data_gts[i]))]

    # res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    # gts_ = {i: gts[i // seq_per_img] for i in range(gen_res_size)}
    # gts_.update({i+gen_res_size: gts[i] for i in range(batch_size)})

    # #pdb.set_trace()

    # avg_scores, cider_scores = CiderD_scorer.compute_score(gts_, res_)
    pdb.set_trace()
    gen_hyp = list_to_string(gen_result, batch_size)
    greedy_hyp = list_to_string(greedy_res, batch_size)
    txt_ref = list_to_string(data_gts, batch_size)

    # need seperate bleu score for every sentence "pair" in batch, scroll up in tmux to see format
    scores = []

    gen_bleu = bleu(references=txt_ref, hypotheses=gen_hyp)
    greedy_bleu = bleu(references=txt_ref, hypotheses=greedy_hyp)

    #scores = txt_bleu["bleu4"]
    #pdb.set_trace()
    #scores = scores[:gen_res_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    #scores = scores.reshape(gen_res_size)
    #rewards = np.repeat(scores[:, np.newaxis], max(map(len, gen_result)), 1)
    #rewards = np.repeat(scores[:, np.newaxis], factor, 1)
    #########################################3
    #pdb.set_trace()
    #if rewards.shape[1] != max(map(len, gen_result)):
    #    pdb.set_trace()
    rewards = 0
    return rewards

# Calculates the CIDEr score for a batch of sentences
def calc_batch_cider(hyp, ref):
    #batch_size = config['batch_size']
    #gen_res_size = hyp.shape[0]
    #seq_per_img = gen_res_size // len(ref)

    #hyp_decode = torch_to_list(hyp, vocab_dict, config)
    #gt_decode = torch_to_list(ref, vocab_dict, config)
    #pdb.set_trace()
    res = OrderedDict()
    for i in range(len(hyp)): # Put both captions into ordered dict
        res[i] = [array_to_str(hyp[i][1:])]
    
    gts = OrderedDict() # Put gts into similar format ordered dict
    for i in range(len(ref)):
        gts[i] = [array_to_str(ref[i])]# for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i] for i in range(len(res))}

    #pdb.set_trace()
    avg_scores, cider_scores = CiderD_scorer.compute_score(gts_, res_)

    return avg_scores*100 # used to be *100

def cider(references, hypotheses):
    """
    Calculates CIDEr-d score for batch of sentences.
    """
    res = OrderedDict()
    for i in range(len(references)):
        res[i] = [hypotheses[i]]
    
    gts = OrderedDict()
    for j in range(len(references)):
        gts[j] = [references[j]]
    
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    gts_ = {i: gts[i] for i in range(len(references))}

    #pdb.set_trace()
    avg_scores, cider_scores = CiderD_scorer.compute_score(gts_, res_)

    return avg_scores*100

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward): # (sample_logprobs, gen_result.data, reward)
        #pdb.set_trace() ### DEBUG: this is from self.rl_crit, calculates final gradient expression
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        #pdb.set_trace()
        mask = (seq>1).float() # Purpose: have '1s' where there are actual words and '0s' where there are no words in sequence
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        if (input.shape[0] != reward.shape[0]):
            pdb.set_trace()
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output