#! /usr/bin/env python

import torch
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig

class ElectraClassificationHead(torch.nn.Module):
    def __init__(self):

        super(ElectraClassificationHead, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.dense = torch.nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dense2 = torch.nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(self.electra.config.hidden_size, 1)
        self.gelu = torch.nn.GELU()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraQAMulti(torch.nn.Module):
    def __init__(self):

        super(ElectraQAMulti, self).__init__()

        electra_base = "google/electra-base-discriminator"
        electra_large = "google/electra-large-discriminator"
        self.electra = ElectraModel.from_pretrained(electra_large)
        self.classifier = ElectraClassificationHead()
        self.dropout = torch.nn.Dropout(self.electra.config.hidden_dropout_prob)
        # self.init_weights()

    
    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        verification_logits = torch.sigmoid(self.classifier(sequence_output))

        return verification_logits