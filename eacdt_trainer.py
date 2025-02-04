
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, sampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import copy
import time
import logging
import pickle

from sdt_model import Transformer_Based_Model, MaskedNLLLoss
from sdt_loaders import get_MELD_loaders
from eacdt_model import CLModel, Classifier
from eacl_loss import eacl_loss_function

import logging
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

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

    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size') # 32

    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')
    ### Environment params
    parser.add_argument("--fp16", type=bool, default=False)
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

def get_paramsgroup(model, args, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params

def init_sdt():
    temp = 8
    n_classes = 7
    n_speakers = 9  # MELD
    hidden_dim = 1024  #
    dropout = 0.5  #
    n_head = 8
    D_audio = 300  # MELD
    D_text = 1024  #
    D_visual = 342  # denseface
    sdt_model = Transformer_Based_Model('MELD', temp, D_text, D_visual, D_audio, n_head,
                                        n_classes, hidden_dim, n_speakers, dropout)
    return sdt_model

def init_eacl_with_sdt(args, sdt_model, n_classes):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens("<mask>")
    eacl_model = CLModel(args, sdt_model, n_classes, tokenizer).cuda()
    return eacl_model


def train_or_eval_model(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None,
                        train=False):
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    if args.disable_training_progress_bar:
        pbar = dataloader
    else:
        pbar = tqdm(dataloader)

    for batch_id, data in enumerate(dataloader):

        textf, visuf, acouf, qmask, umask, label, _  = [torch.tensor(x).to(device) for x in data]
        qmask = qmask.permute(1, 0, 2)
        label[~umask.to(bool)] = -1
        lengths = [(umask[i] == 1).sum().item() for i in range(umask.size(0))]

        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                loss, loss_output, log_prob, label_, mask, anchor_scores = _forward(
                    model, loss_function, textf, visuf, acouf,
                    umask, qmask, lengths, label, device
                )
        else:
            loss, loss_output, log_prob, label_, mask, anchor_scores = _forward(
                model, loss_function, textf, visuf, acouf,
                umask, qmask, lengths, label, device
            )

        if args.use_nearest_neighbour:
            pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            pred = torch.argmax(log_prob[mask], dim=-1)

        preds.append(pred)
        labels.append(label_)
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            if batch_id % args.accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            # Если требуется собирать промежуточные представления для анализа
            sentiment_representations.append(loss_output.sentiment_representations)
            sentiment_labels.append(loss_output.sentiment_labels)

    if len(preds) == 0:
        return float('nan'), float('nan'), [], [], float('nan'), [], [] #, [], [], []

    new_preds, new_labels = [], []
    for i, label_tensor in enumerate(labels):
        for j, lbl in enumerate(label_tensor):
            if lbl != -1:
                new_labels.append(lbl.cpu().item())
                new_preds.append(preds[i][j].cpu().item())

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    return_labels = new_labels
    return_preds = new_preds

    # TODO find max_cosine
    max_cosine = loss_output.max_cosine

    # Считаем F1 отдельно по каждому классу
    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)
    f1_scores = []
    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                pred_label.append(1 if new_preds[i] == class_id else 0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                true_label.append(1 if new_labels[i] == class_id else 0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)

    return avg_loss, avg_accuracy, return_labels, return_preds, avg_fscore, f1_scores, max_cosine


def _forward(model, loss_function, textf, visuf, acouf, umask, qmask, lengths, label, device):

    mask = (label != -1)

    if model.training:
        log_prob, masked_mapped_output, _, anchor_scores = model(
            textf, visuf, acouf, umask, qmask, lengths, return_mask_output=True
        )
        loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    else:
        with torch.no_grad():
            log_prob, masked_mapped_output, _, anchor_scores = model(
                textf, visuf, acouf, umask, qmask, lengths, return_mask_output=True
            )
            loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)


    loss = (loss_output.ce_loss * model.args.ce_loss_weight
            + (1 - model.args.ce_loss_weight) * loss_output.cl_loss)

    return loss, loss_output, log_prob, label[mask], mask, anchor_scores

def retrain(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, ce_losses, preds, labels = [], [], [], []
    
    for batch in dataloader:
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                log_prob = model(data)
        else:
            log_prob = model(data) 
        
        loss = loss_function(log_prob, label)
        losses.append(loss.item())
        pred = torch.argmax(log_prob, dim = -1)
        preds.append(pred)
        labels.append(label)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], # [], [], []
        # plot_representations(sentiment_representations, sentiment_labels, sentiment_anchortypes, anchortype_labels)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    f1_scores = []

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    # print(classification_report(new_labels, new_preds, digits=4, target_names=['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']))

    return_labels = new_labels
    return_preds = new_preds

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)
    # list(precision_recall_fscore_support(y_true=new_labels, y_pred=new_preds)[2])

    return avg_loss, avg_ce_loss, avg_accuracy, return_labels, return_preds, avg_fscore, f1_scores

def main():
    args = get_parser()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.anchor_path = './data'
    sdt_model = init_sdt()
    eacl_model = init_eacl_with_sdt(args, sdt_model, 7)

    loss_function = eacl_loss_function # MaskedNLLLoss()
    train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                               batch_size=args.batch_size,
                                                               num_workers=0)

    device = 'cuda:0' if args.cuda else 'cpu'
    eacl_model.f_context_encoder.to(device)
    eacl_model.to(device)

    num_training_steps = 1
    num_warmup_steps = 0
    optimizer = AdamW(get_paramsgroup(eacl_model.module if hasattr(eacl_model, 'module') else eacl_model, args))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
    best_fscore, best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.

    best_model = copy.deepcopy(eacl_model)
    best_test_fscore = 0
    anchor_dist = []

    logger = get_logger(args.save_path + args.dataset_name + '/logging.log')
    for e in range(args.epochs):
        start_time = time.time()


        train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine = \
            train_or_eval_model(eacl_model, loss_function, train_loader, e, device, args, optimizer, lr_scheduler, True)
        lr_scheduler.step()
        # return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine
        valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _ = \
            train_or_eval_model(eacl_model, loss_function, valid_loader, e, device, args)
        test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _ = \
            train_or_eval_model(eacl_model, loss_function, test_loader, e, device, args)
        all_fscore.append([valid_fscore, test_fscore, test_detail_f1])

        logger.info(
            'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                   test_fscore, round(time.time() - start_time, 2)))

        if test_fscore > best_test_fscore:
            best_model = copy.deepcopy(eacl_model)
            best_test_fscore = test_fscore
            torch.save(eacl_model.state_dict(), args.save_path + args.dataset_name + '/model_' + '.pkl')
    
    print('Stage 1 summary')
    print(classification_report(test_label, test_pred, digits=4, target_names=['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']))
    logger.info('finish stage 1 training!')
    
    path = args.save_path

    torch.cuda.empty_cache()
        # laod best 
    with torch.no_grad():
        anchors = eacl_model.map_function(eacl_model.emo_anchor)
        eacl_model.load_state_dict(torch.load(path + args.dataset_name + '/model_' + '.pkl'))
        eacl_model.eval()
        emb_train, emb_val, emb_test = [] ,[] ,[]
        label_train, label_val, label_test = [], [], []
        for batch_id, batch in enumerate(train_loader):
            # input_ids, label = batch
            textf, visuf, acouf, qmask, umask, label, _  = [torch.tensor(x).to(device) for x in batch]
            qmask = qmask.permute(1, 0, 2)
            label[~umask.to(bool)] = -1
            lengths = [(umask[i] == 1).sum().item() for i in range(umask.size(0))]
            # input_orig = input_ids
            input_aug = None
            # input_ids = input_orig.to(device)
            label = label.to(device)
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = eacl_model(input_ids, return_mask_output=True) ### DONT WORKING
            else:
                log_prob, masked_mapped_output, _, anchor_scores = eacl_model(
                    textf, visuf, acouf, umask, qmask, lengths, return_mask_output=True
                ) 
            out_size = masked_mapped_output.size()
            emb_train.append(masked_mapped_output.detach().cpu().view(out_size[0]*out_size[1], out_size[2]))
            label_train.append(label.cpu().view(label.size(0)*label.size(1)))
        emb_train = torch.cat(emb_train, dim=0)
        label_train = torch.cat(label_train, dim=0)
        mask = label_train != -1
        emb_train = emb_train[mask]
        label_train = label_train[mask]
        for batch_id, batch in enumerate(valid_loader):
            # input_ids, label = batch
            # input_orig = input_ids
            textf, visuf, acouf, qmask, umask, label, _  = [torch.tensor(x).to(device) for x in batch]
            qmask = qmask.permute(1, 0, 2)
            label[~umask.to(bool)] = -1
            lengths = [(umask[i] == 1).sum().item() for i in range(umask.size(0))]
            input_aug = None
            # input_ids = input_orig.to(device)
            label = label.to(device)
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = eacl_model(input_ids, return_mask_output=True) ### DONT WORKING
            else:
                log_prob, masked_mapped_output, _, anchor_scores = eacl_model(
                    textf, visuf, acouf, umask, qmask, lengths, return_mask_output=True
                )
            out_size = masked_mapped_output.size() 
            emb_val.append(masked_mapped_output.detach().cpu().view(out_size[0]*out_size[1], out_size[2]))
            label_val.append(label.cpu().view(label.size(0)*label.size(1)))
        emb_val = torch.tensor([]) # torch.cat(emb_val, dim=0)
        label_val = torch.tensor([]) # torch.cat(label_val, dim=0)

        for batch_id, batch in enumerate(test_loader):
            # input_ids, label = batch
            # input_orig = input_ids
            textf, visuf, acouf, qmask, umask, label, _  = [torch.tensor(x).to(device) for x in batch]
            qmask = qmask.permute(1, 0, 2)
            label[~umask.to(bool)] = -1
            lengths = [(umask[i] == 1).sum().item() for i in range(umask.size(0))]
            input_aug = None
            # input_ids = input_orig.to(device)
            label = label.to(device)
            if args.fp16:
                with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = eacl_model(input_ids, return_mask_output=True) ### DONT WORKING
            else:
                log_prob, masked_mapped_output, _, anchor_scores = eacl_model(
                    textf, visuf, acouf, umask, qmask, lengths, return_mask_output=True
                )
            out_size = masked_mapped_output.size() 
            emb_test.append(masked_mapped_output.detach().cpu().view(out_size[0]*out_size[1], out_size[2]))
            label_test.append(label.cpu().view(label.size(0)*label.size(1)))
        emb_test = torch.cat(emb_test, dim=0)
        label_test = torch.cat(label_test, dim=0)
        mask = label_test != -1
        emb_test = emb_test[mask]
        label_test = label_test[mask]

    print("Embedding dataset built")

    all_fscore = []
    trainset = TensorDataset(emb_train, label_train)
    validset = TensorDataset(emb_val, label_val)
    testset = TensorDataset(emb_test, label_test)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=8)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    if args.save_stage_two_cache:
        os.makedirs("cache", exist_ok=True)
        pickle.dump([train_loader, valid_loader, test_loader, anchors], open(f"./cache/{args.dataset_name}.pkl", 'wb'))
    clf = Classifier(args, anchors).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
    best_valid_score = 0
    for e in range(10):
        train_loss, train_ce_loss, train_acc, a, b, train_fscore, train_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), train_loader, e, device, args, optimizer, train=True)
        
        valid_loss, valid_ce_loss,  valid_acc, _, _, valid_fscore, valid_detail_f1  = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), valid_loader, e, device, args, optimizer, train=False)
        test_loss, test_ce_loss,  test_acc, test_label, test_pred, test_fscore, test_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), test_loader, e, device, args, optimizer, train=False)
        
        logger.info( 'Epoch: {}, train_loss: {}, train_ce_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_ce_loss:{}, test_acc: {}, test_fscore: {}'. \
                format(e + 1, train_loss, train_ce_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_ce_loss, test_acc, test_fscore))
        all_fscore.append([valid_fscore, test_fscore])
        if test_fscore > best_valid_score:
            best_valid_score = test_fscore
            # import pickle
            # pickle.dump((test_label, test_pred), open('with_' * str(args.angle_loss_weight) + 'angle_iemocap.pkl', 'wb'))
            torch.save(clf.state_dict(), path + args.dataset_name + '/clf_' + '.pkl')
            f = test_detail_f1
    
    print('Stage 2 summary')
    print(classification_report(test_label, test_pred, digits=4, target_names=['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']))

    all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
    logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
    logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
    logger.info(f)

    all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)






if __name__ == '__main__':
    main()
