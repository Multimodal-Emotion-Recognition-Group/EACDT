import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F


class CLModel(nn.Module):
    def __init__(self, args, contex_encoder, n_classes,tokenizer=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = n_classes
        self.pad_value = args.pad_value
        self.mask_value = 50265
        self.f_context_encoder = contex_encoder

        #num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        self.dim = self.f_context_encoder.hidden_dim

        self.eps = 1e-8
        self.device = "cuda" if self.args.cuda else "cpu"
        self.predictor = nn.Sequential(
            # nn.Linear(self.dim, self.dim),
            # nn.ReLU(),
            nn.Linear(self.dim, self.num_classes)
        )
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)

        self.tokenizer = tokenizer

        if args.dataset_name == "IEMOCAP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5]).to(self.device)
        elif args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "EmoryNLP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/emorynlp_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6]).to(self.device)

    def device(self):
        return self.f_context_encoder.device

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + self.eps

    def _forward(self, textf, visuf, acouf, umask, qmask, lengths):

        sdt_outputs = self.f_context_encoder(textf, visuf, acouf, umask, qmask, lengths)
        # mask_outputs_list = []
        # for i, seq_len in enumerate(lengths):
        #     mask_outputs_list.append(sdt_outputs[i, seq_len-1, :].unsqueeze(0))
        # mask_outputs = torch.cat(mask_outputs_list, dim=0)
        mask_mapped_outputs = self.map_function(sdt_outputs)

        feature = torch.dropout(sdt_outputs, self.dropout, train=self.training)
        feature = self.predictor(feature)

        if self.args.use_nearest_neighbour:
            anchors = self.map_function(self.emo_anchor)
            self.last_emo_anchor = anchors
            anchor_scores = self.score_func(
                mask_mapped_outputs.unsqueeze(2),
                anchors.unsqueeze(0).unsqueeze(0)
            )
        else:
            anchor_scores = None

        return feature, mask_mapped_outputs, sdt_outputs, anchor_scores

    def forward(self, textf, visuf, acouf, qmask, umask, lengths, return_mask_output=False):
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(
            textf, visuf, acouf, qmask, umask, lengths
        )

        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        else:
            return feature


class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors)
        self.args = args

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1)) / 2 + 1e-8

    def forward(self, emb):
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp
