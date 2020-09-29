# Set of utility functions for project
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

# weight initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Time logging
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Experiment: replaces EOS token (3) with a padding token (0)
#             works with variable length sequences and batches
def trg_trim(trg):
    # Determine batch size
    #pdb.set_trace()
    batch_size = trg.shape[0]
    sent_length = trg.shape[1]
    # Iterate over # of sentences
    for i in range(batch_size):
        for j in range(sent_length):
            if trg[i][j] == 3:
                trg[i][j] = 0
                break

    return trg[:, :-1]

# #Takes model output and replaces all indices after EOS token (3) with buffer token (0)
# def sc_trim(out_sent, out_probs):
#     batch_size = out_sent.shape[0]

#     for i in range(batch_size):
#         EOS_track = 0
#         index_track = 0
#         for j in range(out_sent[i].shape[0]): # Just 29
#             if EOS_track == 1:
#                 out_sent[i][j] = 0
#                 out_probs[i][j] = 0
#             else:
#                 out_probs[i][j] = F.log_softmax(out_probs[i][j])
#             if out_sent[i][j].item() == 3:
#                 EOS_track = 1
#                 index_track = j
#                 out_probs[i][j] = F.log_softmax(out_probs[i][j])

#     return out_probs

#Takes model output and replaces all indices after EOS token (3) with buffer token (0)
def sc_trim(out_sent, out_probs):
    batch_size = out_sent.shape[0]
    new_probs = torch.zeros_like(out_probs)

    for i in range(batch_size):
        EOS_track = 0
        index_track = 0
        for j in range(out_sent[i].shape[0]): # Just 29
            if EOS_track == 1:
                out_sent[i][j] = 1
                #out_probs[i][j] = 0
            else:
                new_probs[i][j] = F.log_softmax(out_probs[i][j])
            if out_sent[i][j].item() == 3:
                EOS_track = 1
                index_track = j
                new_probs[i][j] = F.log_softmax(out_probs[i][j])

    return new_probs

# Takes path to line separated vocab file and returns dict 
# with line number as key and vocab word as value
def init_decode(vocab):
    vocab_dictionary = {}

    infile = open(vocab, 'r')
    line_num = 1
    for line in infile:
        vocab_dictionary[line_num] = line.strip()
        line_num += 1
    infile.close()

    return vocab_dictionary

# Takes list(s) of word indices, and vocab_dictionary and returns list(s) of words
# Only used in conjunction with torch_to_list()
def decode_list(seq, vocab_dict, config):
    out = []
    #pdb.set_trace()
    if (config['batch_size'] == 1):
        for idx in seq:
            word = vocab_dict.get(idx)
            out.append(word)
    else: # batch_size > 1
        for i in range(len(seq)): # number of batches
            out.append([])
            for idx in seq[i]: # number of words per seq
                word = vocab_dict.get(idx)
                out[i].append(word)

    return out

# Takes a variable length tensor with word indices and returns a sentence
def torch_to_list(seq, vocab_dict, config):
    #pdb.set_trace()
    temp_list = seq.squeeze(0).tolist()
    out = []
    if (config['batch_size'] == 1):# case for batch size of 1, any larger batch size len(hyp) returns batch size
            for i in range(len(temp_list)):
                if temp_list[i] not in {0, 1, 2}: # 0 = padding token, 1 
                    out.append(temp_list[i])
    else: # batch size > 1
            for i in range(len(temp_list)): # number of batches
                out.append([])
                for j in range(len(temp_list[i])): # number of words per seq
                    if temp_list[i][j] not in {0, 1, 2}:
                        out[i].append(temp_list[i][j])
                        if temp_list[i][j] == 3: # New logic to only translate sentence up until first EOS token
                            break
    out = decode_list(out, vocab_dict, config)
    return out

# Takes a list of words, or a list of list of words and converts them to sentences
def list_to_string(seq, config):
    if config['batch_size'] == 1:
        out =  [" ".join(seq)]
    else:
        out = []
        for i in range(len(seq)):
            out.append([])
            out[i] = " ".join(seq[i])
    
    return out

# Specifically for CIDEr func.
def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

# Takes a dictionary of BLEU scores and the number of samples and returns an averaged dictionary
def average_scores(logs, length, desc):
    if len(desc) == 15: #(Training)  
        logs['bleu_scores/training/bleu_1'] = logs['bleu_scores/training/bleu_1']/length
        logs['bleu_scores/training/bleu_2'] = logs['bleu_scores/training/bleu_2']/length
        logs['bleu_scores/training/bleu_3'] = logs['bleu_scores/training/bleu_3']/length
        logs['bleu_scores/training/bleu_4'] = logs['bleu_scores/training/bleu_4']/length
        logs['cider_scores/training'] = logs['cider_scores/training']/length
    else:
        logs['bleu_scores/validation/bleu_1'] = logs['bleu_scores/validation/bleu_1']/length
        logs['bleu_scores/validation/bleu_2'] = logs['bleu_scores/validation/bleu_2']/length
        logs['bleu_scores/validation/bleu_3'] = logs['bleu_scores/validation/bleu_3']/length
        logs['bleu_scores/validation/bleu_4'] = logs['bleu_scores/validation/bleu_4']/length
        logs['cider_scores/validation'] = logs['cider_scores/validation']/length

    return logs
    