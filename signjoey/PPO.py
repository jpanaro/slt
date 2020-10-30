import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random
import pdb

from trl.core import (logprobs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix)

from signjoey.external_metrics import sacrebleu
from signjoey.helpers import array_to_str, score_conv, filter_logits

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    We will modify this trainer to return a singular loss value per batch
    Optimal batch size will be 256 with 16 forward batch size. (play by ear)
    """
    
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True, 
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256, # Always make sure this batch size matches config batch size, default 256
        "forward_batch_size": 16, # TODO abstract this config to TrainManager, default 16
        "ppo_epochs": 4,    
    } 
    
    def __init__(self, model, ref_model, old_optimizer, **ppo_params):
        """
        Initialize PPOTrainer.
        
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
                
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        
        self.ref_model = ref_model
        #self.ref_model.eval()
        self.model = model
        #self.model.eval()
        #self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])
        self.optimizer = old_optimizer
     
        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])


    def step(self, batch):
        """
        Run a PPO optimisation step.
        
        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]
            
        returns:
            train_stats (dict): a summary of the training statistics
        """
        #pdb.set_trace()
        bs = self.ppo_params['batch_size']
        timing = dict()
        logs = dict()
        t0 = time.time()

        #pdb.set_trace()
        self.model.eval()
        self.ref_model.eval()
        
        #gen_len = response.shape[1]
        #model_input = torch.cat((query, response), axis=1)
        
        t = time.time()
        logprobs, ref_logprobs, values, scores = self.batched_forward_pass(batch)
        scores = torch.FloatTensor(scores).cuda()
        #pdb.set_trace()
        timing['time/ppo/forward_pass'] = time.time()-t

        # calculate scores here (maybe not)
        #active_text = logprobs.argmax(-1)

        t = time.time()
        pdb.set_trace()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        #pdb.set_trace()
        timing['time/ppo/compute_rewards'] = time.time()-t 
        
        t = time.time() 
        all_stats = []
        idxs = list(range(bs))
        #pdb.set_trace()
        for _ in range(self.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            #pdb.set_trace()
            for i in range(bs):
                idx = idxs[i]
                if batch.sgn[idx:idx+1].shape[0] != 0:
                    train_stats = self.train_minibatch(logprobs[idx:idx+1], values[idx:idx+1],
                                                    rewards[idx:idx+1], batch, idx)
                    all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t
        
        t = time.time()
        train_stats = stack_dicts(all_stats)
        #pdb.set_trace()
        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        
        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        # Record reward mean
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        stats.update(timing)
        stats.update(logs)
        return stats

    def batched_forward_pass(self, batch):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []
        scores = []
        #pdb.set_trace()
        for i in range(int(self.ppo_params['batch_size']/fbs)): # 16 times (using 256 bs and 16 fbs)
            m_input = batch.txt_input[i*fbs:(i+1)*fbs] # splits batch into chunks of 16 until 256 is reached
            if batch.sgn[i*fbs:(i+1)*fbs].shape[0] == 0: # Some batches had nothing in them, need to skip those
                break
            active_decoder_outputs, _, active_values = self.model.forward(
                sgn=batch.sgn[i*fbs:(i+1)*fbs],
                sgn_mask=batch.sgn_mask[i*fbs:(i+1)*fbs],
                sgn_lengths=batch.sgn_lengths[i*fbs:(i+1)*fbs],
                txt_input=batch.txt_input[i*fbs:(i+1)*fbs],
                txt_mask=batch.txt_mask[i*fbs:(i+1)*fbs],
            )
            ref_decoder_outputs, _, _ = self.ref_model.forward(
                sgn=batch.sgn[i*fbs:(i+1)*fbs],
                sgn_mask=batch.sgn_mask[i*fbs:(i+1)*fbs],
                sgn_lengths=batch.sgn_lengths[i*fbs:(i+1)*fbs],
                txt_input=batch.txt_input[i*fbs:(i+1)*fbs],
                txt_mask=batch.txt_mask[i*fbs:(i+1)*fbs],
            )
            # Get logits
            #pdb.set_trace()
            logits, _, _, _ = active_decoder_outputs
            ref_logits, _, _, _ = ref_decoder_outputs
            # Filter the logits (needs to be done in decoder probably)
            #pdb.set_trace()
            filter_logits(logits)
            filter_logits(ref_logits)

            # translate each sentence to german
            #pdb.set_trace()
            active_res = logits[:,:-1,:].argmax(2) # Maybe do respond_to_batch() from iwerratrl here
            #reference_res = ref_logits[:,:-1,:].argmax(2)
            active_sentences = self.model.txt_vocab.arrays_to_sentences(arrays=active_res)
            #reference_sentences = self.ref_model.txt_vocab.arrays_to_sentences(arrays=reference_res)
            gt_sentences = self.model.txt_vocab.arrays_to_sentences(arrays=m_input[:,1:])
            # append a bleu-4 score for each sample to the scores array
            for j in range(len(active_sentences)):
                #pdb.set_trace()
                active_temp = array_to_str(active_sentences[j])
                #reference_temp = array_to_str(reference_sentences[j])
                gt_temp = array_to_str(gt_sentences[j])
                #ref_bleu_temp = sacrebleu.sentence_bleu(reference_temp, gt_temp)
                bleu_temp = sacrebleu.sentence_bleu(active_temp, gt_temp)
                #bleu_temp = sacrebleu.sentence_bleu(gt_temp, gt_temp)
                #pdb.set_trace()
                scores.append(score_conv(bleu_temp.scores[3]))
                #ref_scores.append(ref_bleu_temp.scores[3])
            pdb.set_trace()
            values.append(active_values[:,:-1].detach())
            logprobs.append(logprobs_from_logits(logits[:,:-1,:], m_input[:,1:]).detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:,:-1,:], m_input[:,1:]).detach())
        #pdb.set_trace()
        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values), scores
    
    def train_minibatch(self, logprobs, values, rewards, batch, idx):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, batch, idx)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats
    
    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        pdb.set_trace()
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores
        return rewards, non_score_reward, self.kl_ctl.value

    def loss(self, old_logprobs, values, rewards, batch, idx):
        """Calculate policy and value losses."""
        #pdb.set_trace()
        lastgaelam = 0
        advantages_reversed = []
        model_input = batch.txt_input[idx:idx+1]
        gen_len = batch.txt_input[idx:idx+1].shape[1]-1
        
        ###################################################
        # Potentially filter logprobs here.
        ###################################################
        
        #pdb.set_trace()
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        #pdb.set_trace()
        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        #logits, _, vpred = self.model(model_input)
        active_decoder_outputs, _, vpred = self.model.forward(
                sgn=batch.sgn[idx:idx+1],
                sgn_mask=batch.sgn_mask[idx:idx+1],
                sgn_lengths=batch.sgn_lengths[idx:idx+1],
                txt_input=batch.txt_input[idx:idx+1],
                txt_mask=batch.txt_mask[idx:idx+1],
            )
        logits, _, _, _ = active_decoder_outputs
        filter_logits(logits)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
        
        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)
        
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())
        
        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats