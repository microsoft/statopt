# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from .qhm import QHM
from .bucket import LeakyBucket


class SLOPE(QHM):
    r"""
    Statistical Adaptive Slope Test. (t-test)

    optimizer = SLOPE(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0,
                     warmup=0, drop_factor=2, significance=0.02, var_mode='linear',
                     leak_ratio=8, minN_stats=400, testfreq=100, logstats=0)

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):

        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k)
        x(k+1) = x(k) - \alpha * d(k)

    Stationary criterion:
        H0: slope >= 0 vs H1: slope < 0

    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range(0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range(0,1) (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum and dampened gradient
            \nu = \beta: SGD with "Nesterov momentum"
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        warmup (int, optional): number of steps before testing (default: 100)
        dropfactor (float, optional): factor of drop learning rate (default: 2)
        significance (float, optional): test significance level (default:0.02)
        var_mode (string, optional): linear regression mode (default: 'linear')
        leak_ratio (int, optional): leaky bucket ratio to kept (default: 8)
        minN_stats (int, optional): min number of samples for test (default: 400)
        testfreq (int, optional): number of steps between testing (default:100)
        logstats (int, optional): number of steps between logs (0 means no log)

    Example:
        >>> optimizer = torch.optim.SLOPE(model.parameters(), lr=0.1, momentum=0.9, 
                                          weight_decay=0.0005)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0,
                 warmup=100, drop_factor=2, significance=0.05, var_mode='linear',
                 leak_ratio=8, minN_stats=1000, testfreq=100, logstats=0):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if drop_factor < 1:
            raise ValueError("Invalid value for drop_factor (>=1): {}".format(drop_factor))
        if significance <= 0 or significance >= 1:
            raise ValueError("Invalid value for significance (0,1): {}".format(significance))
        if var_mode not in ['linear', 'log']:
            raise ValueError("Invalid value for var_mode ('linear', 'log'): {}".format(var_mode))
        if leak_ratio < 1:
            raise ValueError("Invalid value for leak_ratio (int, >=1): {}".format(leak_ratio))
        if minN_stats < 100:
            raise ValueError("Invalid value for minN_stats (int, >=100): {}".format(minN_stats))
        if warmup < 0:
            raise ValueError("Invalid value for warmup (int, >1): {}".format(warmup))
        if testfreq < 1:
            raise ValueError("Invalid value for testfreq (int, >=1): {}".format(testfreq))

        super(SLOPE, self).__init__(params, lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = LeakyBucket(1000, leak_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['drop_factor'] = drop_factor
        self.state['significance'] = significance
        self.state['var_mode'] = var_mode
        self.state['minN_stats'] = int(minN_stats)
        self.state['warmup'] = int(warmup)
        self.state['testfreq'] = int(testfreq)
        self.state['logstats'] = int(logstats)
        self.state['nSteps'] = 0

        # statistics to monitor
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['stats_mean'] = 0
        self.state['conf'] = 0
        self.state['slope'] = 0

    def step(self, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = closure()

        self.add_weight_decay()
        self.qhm_direction()
        self.qhm_update()
        self.state['nSteps'] += 1
        self.stats_adaptation(loss)

        return loss

    def stats_adaptation(self, loss):
        # add loss statistic to leaky bucket
        bucket = self.state['bucket']
        bucket.add(loss.item() + self.L2_regu_loss())

        # check statistics and adjust learning rate
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        if bucket.count > self.state['minN_stats'] and self.state['nSteps'] % self.state['testfreq'] == 0:
            is_decreasing, slope, prob = bucket.linregress(self.state['significance'],
                                                           self.state['var_mode'])
            self.state['stats_test'] = 1
            self.state['stats_stationary'] = 1 - int(is_decreasing)
            self.state['conf'] = prob
            self.state['slope'] = slope
            # perform statistical test for stationarity
            if self.state['nSteps'] > self.state['warmup'] and not is_decreasing:
                self.state['lr'] /= self.state['drop_factor']
                for group in self.param_groups:
                    group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer')
                bucket.reset()

        # Log statistics only for debugging. Therefore self.state['stats_test'] remains False
        if self.state['logstats'] and not self.state['stats_test']:
            if bucket.count > bucket.ratio and self.state['nSteps'] % self.state['logstats'] == 0:
                is_decreasing, slope, prob = bucket.linregress(self.state['significance'],
                                                             self.state['var_mode'])
                self.state['stats_stationary'] = 1 - int(is_decreasing)
                self.state['conf'] = prob
                self.state['slope'] = slope

 
    def _zero_buffers(self, buf_name):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    state[buf_name].zero_()
        return None

 
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
