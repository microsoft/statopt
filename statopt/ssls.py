# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch.optim import Optimizer
from .qhm import QHM


class SSLS(QHM):
    r"""
    QHM with Smoothed Stochastic Line Search (SSLS) for tuning learning rates

    optimizer = SSLS(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, gamma=0.1, 
                     ls_sdc=0.1, ls_inc=2.0, ls_dec=0.5, ls_max=10, 
                     ls_evl=1, ls_dir='g', ls_cos=0)

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha(k) * d(k)   

    where \alpha(k) is smoothed version of \eta(k) obtained by line search 
    (line search performed on loss defined by current mini-batch)

        \alpha(k) = (1 - \gamma) * \alpha(k-1) + \gamma * \eta(k)
    
    Suggestion: set smoothing parameter by batch size:  \gamma = a * b / n
    The cumulative increase or decrease efficiency per epoch is (1-exp(-a)) 
 
    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range(0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range(0,1) (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum and dampened gradient
            \nu = \beta: SGD with "Nesterov momentum"
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
        gamma (float, optional): smoothing parameter for line search (default: 0.01)
    The next four arguments can be tuned, but defaults should work well.
        ls_sdc (float, optional): sufficient decreasing coefficient (default: 0.05)
        ls_inc (float, optional): incremental factor (>1, default: 2.0)
        ls_dec (float, optional): decremental factor (<1, default: 0.5)
        ls_max (int, optional): maximum number of line searches (default: 2)
    The next three arguments are for research purpose only!
        ls_evl (bool, optional): whether or not use evaluation mode (default: 1)
        ls_dir (char, optional): 'g' for g(k) and 'd' for d(k) (default: 'g')
        ls_cos (bool, optional): whether or not use cosine between g and d (default: 0)

    How to use it:
    >>> optimizer = SSLS(model.parameters(), lr=1, momentum=0.9, qhm_nu=1,
    >>>                  weight_decay=1e-4, gamma=0.01)
    >>> for input, target in dataset:
    >>>     model.train()
    >>>     optimizer.zero_grad()
    >>>     loss_func(model(input), target).backward()
    >>>     def eval_loss(eval_mode=True):  # closure function for line search
    >>>         if eval_mode:
    >>>             model.eval()
    >>>         with torch.no_grad():
    >>>             output = model(input)
    >>>             loss = loss_func(output, target)
    >>>         return loss
    >>>     optimizer.step(eval_loss)
    """

    def __init__(self, params, lr=1e-3, momentum=0, qhm_nu=1, weight_decay=0, gamma=0.1, 
                 ls_sdc=0.05, ls_inc=2.0, ls_dec=0.5, ls_max=2, 
                 ls_evl=1, ls_dir='g', ls_cos=0):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>=0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1]: {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if gamma < 0 or gamma > 1:
            raise ValueError("Invalid value for gamma [0,1]: {}".format(gamma))
        if ls_sdc <= 0 or ls_sdc >= 0.5:
            raise ValueError("Invalid value for ls_sdc (0,0.5): {}".format(ls_sdc))
        if ls_inc < 1 :
            raise ValueError("Invalid value for ls_inc (>=1): {}".format(ls_inc))
        if ls_dec <= 0 or ls_dec >= 1:
            raise ValueError("Invalid value for ls_dec (0,1): {}".format(ls_dec))
        if ls_max < 1:
            raise ValueError("Invalid value for ls_max (>=1): {}".format(ls_max))
        if ls_dir not in['g', 'd']:
            raise ValueError("Invalid value for ls_dir ('g' or 'd'): {}".format(ls_dir))

        super(SSLS, self).__init__(params, lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        # Extra_buffer used only if momentum > 0 and nu != 1 even though True is declared here!
        self.state['allocate_step_buffer'] = True

        self.state['lr'] = float(lr)
        self.state['gamma'] = gamma
        self.state['ls_sdc'] = ls_sdc
        self.state['ls_inc'] = ls_inc
        self.state['ls_dec'] = ls_dec
        self.state['ls_max'] = int(ls_max)
        self.state['ls_evl'] = bool(ls_evl)
        self.state['ls_dir'] = ls_dir
        self.state['ls_cos'] = bool(ls_cos)

        # state for tracking cosine of angle between g and d, line search result and count.
        self.state['cosine'] = 0.0
        self.state['ls_eta'] = 0.0
        self.state['ls_cnt'] = 0

    def step(self, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, eval_mode): A closure that reevaluates model and returns loss.
        """

        self.add_weight_decay()
        self.qhm_direction()
        loss, _ = self.line_search(closure)
        self.qhm_update()

        return loss

    def line_search(self, closure):
        # need loss values using evaluation mode (or not)
        loss0 = closure(self.state['ls_evl'])

        # QHM search direction should be determined before line search and update
        g_dot_d = 0.0
        g_norm2 = 0.0
        d_norm2 = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # first make copy of parameters before doing line search
                if 'ls_buffer' not in state:
                    state['ls_buffer'] = torch.zeros_like(p.data)
                state['ls_buffer'].copy_(p.data)

                # compute inner product between g and d
                g = p.grad.data.view(-1)
                # should use (g.dot(d)).item() to use scalars!
                if self.state['ls_dir'] == 'g':
                    d = g
                else: 
                    d = state['step_buffer'].view(-1)
                g_dot_d += g.dot(d).item()

                # if self.state['ls_dir'] == 'd' and self.state['ls_cos']:
                g_norm2 += g.dot(g).item()
                d_norm2 += d.dot(d).item()

        # line search on current mini-batch (not changing input to model)
        f0 = loss0.item() + self.L2_regu_loss()
        self.state['cosine'] = g_dot_d / math.sqrt(g_norm2 * d_norm2) 
        # try a large instantaneous step size at beginning of line search
        if self.state['ls_dir'] == 'd' and self.state['ls_cos']:
            # The following also decreases eta from lr if cosine < 0
            # self.state['cosine'] = g_dot_d / math.sqrt(g_norm2 * d_norm2) 
            eta = self.state['lr'] * math.pow(self.state['ls_inc'], self.state['cosine'])
        else:
            eta = self.state['lr'] * self.state['ls_inc']

        ls_count = 0
        #while ls_count < self.state['ls_max']:
        while g_dot_d > 0 and ls_count < self.state['ls_max']:
            # update parameters x := x - eta * d
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if ls_count > 0:
                        p.data.copy_(self.state[p]['ls_buffer'])
                    if self.state['ls_dir'] == 'g':
                        p.data.add_(-eta, p.grad.data)
                    else:
                        p.data.add_(-eta, self.state[p]['step_buffer'])

            # evaluate loss of new parameters
            f1 = closure(self.state['ls_evl']).item() + self.L2_regu_loss()
            # back-tracking line search
            if f1 > f0 - self.state['ls_sdc'] * eta * g_dot_d:
                eta *= self.state['ls_dec']
            # Goldstein line search: not effective in increasing learning rate
            # elif f1 < f0 - (1 - self.state['ls_sdc']) * eta * g_dot_d:
            #     eta *= self.state['ls_inc']
            else:
                break
            ls_count += 1
        else:
            if g_dot_d <=0 and not self.state['ls_cos']:  
            # if g_dot_d <=0:  
                eta = self.state['lr'] * self.state['ls_dec']
            #if g_dot_d > 0, then result of while loop is eta = lr * power(ls_dec, ls_max)

        # After line search over instantaneous step size, update learning rate by smoothing
        self.state['ls_eta'] = eta
        self.state['ls_cnt'] = ls_count
        self.state['lr'] = (1 - self.state['gamma']) * self.state['lr'] + self.state['gamma'] * eta
        # update lr in parameter groups AND reset weights to original value before line search
        for group in self.param_groups:
            group['lr'] = self.state['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data.copy_(self.state[p]['ls_buffer'])

        # f0 is always computed, but f1 may not. 
        return loss0, f0

    def L2_regu_loss(self):
        L2_loss = 0.0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data.view(-1)
                L2_loss += 0.5 * weight_decay * (x.dot(x)).item()
        return L2_loss

    def gradient_norm(self):
        normsqrd = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data.view(-1)
                normsqrd += (g.dot(g)).item()
        return math.sqrt(normsqrd)

    def buffer_norm(self, buf_name):
        normsqrd = 0.0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    v = state[buf_name].data.view(-1)
                    normsqrd += v.dot(v)
        return math.sqrt(normsqrd)
