import os 
import sys
import json


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from src.models.classifier import *
from src.models.bert_basic_model import *
from src.models.bert_layernorm import BertLayerNorm


class BertClassifier(nn.Module):
    def __init__(self, config, num_labels=4):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        print(config.config_path)
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert = BertModel(bert_config)
        bert_frozen = "false"
        if bert_frozen == "true":
            print("!-!" * 20)
            print("Please notice that the bert grad is false")
            print("!-!" * 20)
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier1 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        # self.classifier2 = nn.Linear(int(config.hidden_size/2), num_labels)

        # self.layer_norm = BertLayerNorm(config.hidden_size, )

        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError
        self.bert = self.bert.from_pretrained(config.bert_model)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, \
                labels=None, input_mask=None):
        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, \
                                                   output_all_encoded_layers=False)

        last_bert_layer = last_bert_layer[:, 0, :]

        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.classifier(last_bert_layer)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if input_mask is not None:
                # print("-*-"*10)
                # print("start using the mask of the sent")
                # masked_logits.masked_scatter_(input_mask, logits)
                masked_logits = torch.masked_select(logits, input_mask)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits