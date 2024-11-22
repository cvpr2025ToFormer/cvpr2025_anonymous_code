from . import BaseLoss
import torch

class L1L2Loss(BaseLoss):
    def __init__(self, args):
        super(L1L2Loss, self).__init__(args)

        self.loss_name = []
        self.t_valid = 0.0001
        self.args = args

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output, epoch):
        loss_val = []

        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            if self.args.model == 'ToFormer':
                pred = output['pred']
                pred_2 = output['pred_2']
                pred_4 = output['pred_4']
                gt = sample['gt']
                gt_2 = sample['gt_2']
                gt_4 = sample['gt_4']

                loss_tmp = 0.0
                a_s1 = 0.80
                a_s2 = 0.16
                a_s4 = 0.04
                if epoch >= 0 and epoch <= 60:
                    a_s1 = 0.80
                    a_s2 = 0.16
                    a_s4 = 0.04
                elif epoch > 60 and epoch <= 75:
                    a_s1 = 0.90
                    a_s2 = 0.08
                    a_s4 = 0.02
                elif epoch > 75:
                    a_s1 = 1.00
                    a_s2 = 0.0
                    a_s4 = 0.0

                if loss_type in ['L1', 'L2']:
                    loss_tmp += loss_func(pred, gt) * a_s1
                    loss_tmp += loss_func(pred_2, gt_2) * a_s2
                    loss_tmp += loss_func(pred_4, gt_4) * a_s4
                else:
                    raise NotImplementedError

            else:
                pred = output['pred']
                gt = sample['gt']
                loss_tmp = 0.0
                if loss_type in ['L1', 'L2']:
                    loss_tmp += loss_func(pred, gt) * 1.0
                else:
                    raise NotImplementedError

            loss_tmp = loss['weight'] * loss_tmp
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)

        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val
