# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from scipy import stats
import torch
import torch.nn.functional as F

# Use a Leaky Bucket to store a fraction of most recent statistics 
# Wikepedia article: https://en.wikipedia.org/wiki/Leaky_bucket
class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 
               2 - keep most recent 1/2, 
               3 - keep most recent 1/3,
               ... 
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        '''
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        newbuffer = torch.zeros(self.size * 2, dtype=self.buffer.dtype, device=self.buffer.device)
        newbuffer[0:self.size][:] = self.buffer
        self.buffer = newbuffer
        self.size *= 2

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            if self.start < self.ratio:             # if the start index is small
                self.double_size()                      # double the size of buffer
            else:                                   # otherwise shift the queue
                self.buffer[0:self.count] = self.buffer[self.start:self.end] 
                self.start = 0                          # reset start index to 0
                self.end = self.count                   # reset end index to count

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1                         # increase count in queue by one
            else:                                   # otherwise leak and keep same count
                self.start += 1                         # increase start index by one

        self.total_count += 1                   # always increase total_count by one

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs).item()
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof

    def stats_test(self, sigma, mode='bm', composite_test=False):
        mean, std, dof = self.mean_std(mode=mode)

        # confidence interval
        t_sigma_dof = stats.t.ppf(1-sigma/2., dof)
        half_width = std * t_sigma_dof / math.sqrt(self.count)
        lower = mean - half_width
        upper = mean + half_width
        # The simple confidence interval test    
        # stationarity = lower < 0 and upper > 0

        # A more stable test is to also check if two half-means are of the same sign
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        stationarity = (lower < 0 and upper > 0) and (mean1 * mean2 > 0)

        if composite_test:
            # Use two half tests to avoid false positive caused by crossing 0 in transient phase
            lb1 = mean1 - half_width
            ub1 = mean1 + half_width
            lb2 = mean2 - half_width
            ub2 = mean2 + half_width
            stationarity = (lb1 * ub1 < 0) and (lb2 * ub2 < 0) and (mean1 * mean2 > 0)

        return stationarity, mean, lower, upper  

    # method to test if average loss after line search is no longer decreasing 
    def rel_reduction(self):
        if self.count < 4:
            return 0.5
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1
        
    # method to test if average loss after line search is no longer decreasing 
    def is_decreasing(self, min_cnt=1000, dec_rate=0.01):
        if self.count < min_cnt:
            return True
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1 > dec_rate

    def linregress(self, sigma, mode='linear'):
        """
        calculate a linear regression
        sigma: the confidence of the one-side test
            H0: slope >= 0 vs H1: slope < 0
        mode: whether log scale the x axis
        """
        TINY = 1.0e-20
        x = torch.arange(self.total_count-self.count, self.total_count,
                         dtype=self.buffer.dtype, device=self.buffer.device)
        if mode == 'log':
            x = torch.log(x)
        # both x and y has dimension (self.count,)
        xy = torch.cat([x.view(1, -1),
                        self.buffer[self.start:self.end].view(1, -1)],
                       dim=0)
        # compute covariance matrix
        fact = 1.0 / self.count
        xy -= torch.mean(xy, dim=1, keepdim=True)
        xyt = xy.t()
        cov = fact * xy.matmul(xyt).squeeze()
        # compute the t-statistics
        r_num = cov[0, 1].item()
        r_den = torch.sqrt(cov[0, 0]*cov[1, 1]).item()
        if r_den == 0.0:
            r = 0.0
        else:
            r = r_num / r_den
            # test for numerical error propagation
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0

        df = self.count - 2
        t = r * math.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        # one-sided test for decreasing
        prob = stats.t.cdf(t, df)
        is_decreasing = prob < sigma
        # slop
        slope = r_num / cov[0, 0].item()
        return is_decreasing, slope, prob
