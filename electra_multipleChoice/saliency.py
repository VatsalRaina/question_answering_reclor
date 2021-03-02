#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import time
import datetime

from transformers import ElectraTokenizer, ElectraForMultipleChoice
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig


import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

MAXLEN = 256

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predicted values')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    with open(args.test_data_path) as f:
        test_data = json.load(f)

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    item = test_data[0]

    context = item["context"]
    question = item["question"]
    four_inp_ids = []
    four_tok_type_ids = []
    for i, ans in enumerate(item["answers"]):
        combo = context + " [SEP] " + question + " " + ans
        inp_ids = tokenizer.encode(combo)
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
        four_inp_ids.append(inp_ids)
        four_tok_type_ids.append(tok_type_ids)
    four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
    four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")

    input_ids = [four_inp_ids]
    token_type_ids = [four_tok_type_ids]

    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        sen_attention_masks = []
        for opt in sen:
            att_mask = [int(token_id > 0) for token_id in opt]
            sen_attention_masks.append(att_mask)
        attention_masks.append(sen_attention_masks)
    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)


    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    embedding_matrix = model.electra.embeddings.word_embeddings
    embedded = torch.tensor(embedding_matrix(input_ids), requires_grad=True)
    # print(embedded)
    print(embedded.size())

    outputs = model(inputs_embeds=embedded, attention_mask=attention_masks, token_type_ids=token_type_ids)
    logits = torch.squeeze(outputs[0])
    print(logits.size())

    logit_optA = logits[0]
    print(logit_optA)
    logit_optB = logits[1]
    print(logit_optB)
    logit_optC = logits[2]
    print(logit_optC)
    logit_optD = logits[3]
    print(logit_optD)



    # Get saliency relative to option A prediction
    logts = logit_optA + logit_optB + logit_optC + logit_optD
    logts.backward()

    saliency_max = torch.squeeze(torch.norm(embedded.grad.data.abs(), dim=3))
    print(saliency_max.size())
    saliency_max = saliency_max.detach().cpu().numpy()

    ansA = item["answers"][0]
    wordsA = tokenizer.tokenize(ansA)
    ansB = item["answers"][1]
    wordsB = tokenizer.tokenize(ansB)
    ansC = item["answers"][2]
    wordsC = tokenizer.tokenize(ansC)
    ansD = item["answers"][3]
    wordsD = tokenizer.tokenize(ansD)


    # Get the saliency values for option A specifically
    saliency_maxA = saliency_max[0, :]
    # Get rid of the first and last tokens
    saliency_maxA = saliency_maxA[1:-1]
    # Get saliency across the answer
    saliency_maxA = saliency_maxA[:len(wordsA)]

    # Get the saliency values for option A specifically
    saliency_maxB = saliency_max[1, :]
    # Get rid of the first and last tokens
    saliency_maxB = saliency_maxB[1:-1]
    # Get saliency across the answer
    saliency_maxB = saliency_maxB[:len(wordsB)]

    # Get the saliency values for option A specifically
    saliency_maxC = saliency_max[2, :]
    # Get rid of the first and last tokens
    saliency_maxC = saliency_maxC[1:-1]
    # Get saliency across the answer
    saliency_maxC = saliency_maxC[:len(wordsC)]

    # Get the saliency values for option A specifically
    saliency_maxD = saliency_max[3, :]
    # Get rid of the first and last tokens
    saliency_maxD = saliency_maxD[1:-1]
    # Get saliency across the answer
    saliency_maxD = saliency_maxD[:len(wordsD)]

    # Normalise values across all answer options
    saliency_maxA = saliency_maxA / (np.sum(saliency_maxA)+np.sum(saliency_maxB)+np.sum(saliency_maxC)+np.sum(saliency_maxD))
    saliency_maxB = saliency_maxB / (np.sum(saliency_maxA)+np.sum(saliency_maxB)+np.sum(saliency_maxC)+np.sum(saliency_maxD))
    saliency_maxC = saliency_maxC / (np.sum(saliency_maxA)+np.sum(saliency_maxB)+np.sum(saliency_maxC)+np.sum(saliency_maxD))
    saliency_maxD = saliency_maxD / (np.sum(saliency_maxA)+np.sum(saliency_maxB)+np.sum(saliency_maxC)+np.sum(saliency_maxD))

    print(question)
    print(item["answers"])
    ansA = item["answers"][0]
    combo = context + " [SEP] " + question + " " + ans
    print(combo)
    # words = tokenizer.tokenize(combo)

    M = len(wordsA)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,40))
    plt.barh(xx, list(saliency_maxA)[::-1])
    plt.yticks(xx, labels=np.flip(wordsA), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Option A')
    plt.ylim([-2, M+2])
    plt.xlim([0.0, 0.8])
    plt.savefig('./saliencyA.png')
    plt.close()

    M = len(wordsB)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,40))
    plt.barh(xx, list(saliency_maxB)[::-1])
    plt.yticks(xx, labels=np.flip(wordsB), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Option B')
    plt.ylim([-2, M+2])
    plt.xlim([0.0, 0.8])
    plt.savefig('./saliencyB.png')
    plt.close()

    M = len(wordsC)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,40))
    plt.barh(xx, list(saliency_maxC)[::-1])
    plt.yticks(xx, labels=np.flip(wordsC), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Option C')
    plt.ylim([0, M])
    plt.xlim([0.0, 0.8])
    plt.savefig('./saliencyC.png')
    plt.close()

    M = len(wordsD)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,40))
    plt.barh(xx, list(saliency_maxD)[::-1])
    plt.yticks(xx, labels=np.flip(wordsD), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Option D')
    plt.ylim([-2, M+2])
    plt.xlim([0.0, 0.08])
    plt.savefig('./saliencyD.png')
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)