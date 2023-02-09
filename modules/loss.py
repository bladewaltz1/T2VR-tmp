import torch
import torch.nn as nn
import torch.nn.functional as F

from config.base_config import Config


class CLIPLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device)

    def forward_async(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale

        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

    def forward_sync(self, sims_t2v, sims_v2t, logit_scale):
        """
        Inputs: cosine similarities
            sims_t2v: n x kn
            sims_v2t: n x kn
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits_t2v = sims_t2v * logit_scale
        logits_v2t = sims_v2t * logit_scale
        N = logits_t2v.shape[0]

        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank())
        labels = labels.to(self.device)

        t2v_loss = F.cross_entropy(logits_t2v, labels)
        v2t_loss = F.cross_entropy(logits_v2t, labels)

        return (t2v_loss + v2t_loss) / 2.0

    def forward(self, *args, **kargs):
        if self.training and torch.distributed.get_world_size() > 1:
            return self.forward_sync(*args, **kargs)
        else:
            return self.forward_async(*args, **kargs)


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss(config)
        else:
            raise NotImplemented
