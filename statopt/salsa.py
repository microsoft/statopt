# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .ssls import SSLS
from .sasa import SASA
from .bucket import LeakyBucket


class SALSA(SASA, SSLS):
    r"""
    SALSA: Statistical Approximation with Line-search and Statistical Adaptation
    a combination of the following two methods with automatic switch 
    SSLS: Smoothed Stochastic Line Search
    SASA: Statistical Adaptive Stochastic Approximation

    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha(k) * d(k)   

    How to use it: (same as SSLS, except for a warmup parameter for SASA)
    >>> optimizer = SALSA(model.parameters(), lr=1, momentum=0.9, qhm_nu=1,
    >>>                   weight_decay=1e-4, gamma=0.01, warmup=1000)
    >>> for input, target in dataset:
    >>>     model.train()
    >>>     optimizer.zero_grad()
    >>>     loss_func(model(input), target).backward()
    >>>     def eval_loss(eval_mode=True):    # closure function for line search
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
                 ls_evl=1, ls_dir='g', ls_cos=0,    # should not change these 3 defaults
                 auto_switch=1,
                 warmup=0, drop_factor=10, significance=0.05, var_mode='mb',
                 leak_ratio=8, minN_stats=1000, testfreq=100, logstats=0):

        SASA.__init__(self, params, lr, momentum, qhm_nu, weight_decay, 
                      warmup, drop_factor, significance, var_mode, 
                      leak_ratio, minN_stats, testfreq, logstats)
        # State from QHM: Extra_buffer used only if momentum > 0 and nu != 1
        self.state['allocate_step_buffer'] = True

        # Initialize states of SSLS here       
        self.state['lr'] = float(lr)
        self.state['gamma'] = gamma
        self.state['ls_sdc'] = ls_sdc
        self.state['ls_inc'] = ls_inc
        self.state['ls_dec'] = ls_dec
        self.state['ls_max'] = int(ls_max)
        self.state['ls_evl'] = bool(ls_evl)
        self.state['ls_dir'] = ls_dir
        self.state['ls_cos'] = bool(ls_cos)

        # state for tracking cosine of angle between g and d.
        self.state['cosine'] = 0.0
        self.state['ls_eta'] = 0.0
        self.state['ls_cnt'] = 0

        self.state['auto_switch'] = bool(auto_switch)
        self.state['switched'] = False

        # State initialization: leaky bucket to store mini-batch loss values
        p = self.param_groups[0]['params'][0]
        self.state['ls_bucket'] = LeakyBucket(1000, leak_ratio, p.dtype, p.device)

    def step(self, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates model and returns loss.
        """

        if self.state['auto_switch']:
            self.step_auto_switch(closure)
        else:
            self.step_mannual_switch(closure)

        return None


    def step_auto_switch(self, closure):

        self.add_weight_decay()
        self.qhm_direction()

        if not self.state['switched']:
            _, fval = self.line_search(closure)
            self.state['ls_bucket'].add(fval)
            if self.state['ls_bucket'].count > self.state['minN_stats']:
                is_decreasing = self.state['ls_bucket'].linregress(self.state['significance'])[0]
                if not is_decreasing:
                    self.state['switched'] = True
                    print("SALSA: auto switch due to non-descreasing training loss.")

        self.qhm_update()

        self.state['nSteps'] += 1
        self.stats_adaptation()
        if self.state['stats_test'] and self.state['stats_stationary']:
            self.state['switched'] = True
            print("SALSA: auto switch due to stationarityy test")


    def step_mannual_switch(self, closure):

        if self.state['nSteps'] < self.state['warmup']:
            SSLS.step(self, closure)
            self.state['nSteps'] += 1
        else: 
            SASA.step(self)

