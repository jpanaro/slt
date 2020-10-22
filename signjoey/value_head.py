import math
import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torch import Tensor
import pdb

class ValueHead(nn.Module):
    """
    The ValueHead class implements an extra head for the Transformer model that returns a scalr for each
    output token. This acts as the value function for the PPO model
    """
    def __init__(self, config):
        super().__init__()
        #pdb.set_trace()
        self.detach_head = False
        self.summary_type = config["summary_type"] if "summary_type" in config else "last"

        self.summary = Identity()
        if "summary_use_proj" in config and config["summary_use_proj"]:
            if "summary_proj_to_labels" in config and config["summary_proj_to_labels"] and config["num_labels"] > 0:
                num_classes = config["num_labels"]
            else:
                num_classes = config["hidden_size"]
            self.summary = nn.Linear(config["hidden_size"], num_classes)

        self.activation = Identity()
        if "summary_activation" in config and config["summary_activation"] == "tanh":
            self.activation = nn.Tanh()
        
        self.first_dropout = Identity()
        if "summary_first_dropout" in config and config["summary_first_dropout"] > 0:
            self.first_dropout = nn.Dropout(config["summary_first_dropout"])
        
        self.last_dropout = Identity()
        if "summary_last_dropout" in config and config["summary_last_dropout"] > 0:
            self.last_dropout = nn.Dropout(config["summary_last_dropout"])
        
        self.flatten = nn.Flatten()
    
    def forward(self, hidden_states, cls_index=None):
        if self.detach_head: # Not being used by default
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output