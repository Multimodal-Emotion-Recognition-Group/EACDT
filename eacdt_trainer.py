
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import AdamW

from sdt_model import Transformer_Based_Model, MaskedNLLLoss
from sdt_loaders import get_MELD_loaders
from eacl_model import CLModel, Classifier
from eacl_loss import eacl_loss_function

import logging
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='princeton-nlp/sup-simcse-roberta-large')
    parser.add_argument('--bert_dim', type=int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--mask_value', type=int, default=2, help='padding')
    parser.add_argument('--wp', type=int, default=8, help='past window size')
    parser.add_argument('--wf', type=int, default=0, help='future window size')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=256,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument('--accumulation_step', type=int, default=1)

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='MELD', type=str, help='dataset name, IEMOCAP or MELD or EmoryNLP')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')

    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=8, metavar='E', help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')
    ### Environment params
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ignore_prompt_prefix", action="store_true", default=True)
    parser.add_argument("--disable_training_progress_bar", action="store_true")
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)

    # ablation study
    parser.add_argument("--disable_emo_anchor", action='store_true')
    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--disable_two_stage_training", action="store_true")
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)
    parser.add_argument("--anchor_path", type=str)

    # analysis
    parser.add_argument("--save_stage_two_cache", action="store_true")
    parser.add_argument("--save_path", default='./saved_models/', type=str)

    args = parser.parse_args()
    return args

def init_sdt():
    temp = 8
    n_classes = 7
    n_speakers = 9  # MELD
    hidden_dim = 1024  #
    dropout = 0.5  #
    n_head = 16
    D_audio = 300  # MELD
    D_text = 1024  #
    D_visual = 342  # denseface
    sdt_model = Transformer_Based_Model('MELD', temp, D_text, D_visual, D_audio, n_head,
                                        n_classes, hidden_dim, n_speakers, dropout)
    return sdt_model

def init_eacl(args, n_classes):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens("<mask>")
    eacl_model = CLModel(args, n_classes, tokenizer).cuda()
    return eacl_model

def main():
    args = get_parser()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.anchor_path = './data'
    sdt_model = init_sdt()
    eacl_model = init_eacl(args, 7)

    loss_function = MaskedNLLLoss()
    train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                               batch_size=args.batch_size,
                                                               num_workers=0)

    device = 'cuda:0' if args.cuda else 'cpu'
    sdt_model.to(device)

    for data in train_loader:

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        all_tf_emb = sdt_model(textf, visuf, acouf, umask, qmask, lengths)




if __name__ == '__main__':
    main()
