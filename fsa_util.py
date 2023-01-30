import torch
import numpy as np

def slack_oms(unsquared_upper_n, lower_n, n, tau=0.0):
    """
    Confidence Score Estimation of Out-Performance

    On the Certification of Classifiers for Outperforming Human Annotators (ICLR 2023)
    https://openreview.net/forum?id=X5ZMzRYqUjB

    Arguments:
        unsquared_upper_n: unsquared upper bound (with finite N samples)
        lower_n: lower bound (with finite N samples)
        n: number of samples (N)
        tau: margin tau (default 0.0)
    """
    print('input:', 'tau', tau, 'unsquared_upper', unsquared_upper_n, 'lower_n', lower_n, 'N', n)

    def delta(n, t): # Eqn 16
        return torch.exp(-n *2 * t * t )
    
    def tl_func(tu): # Eqn 19
        return lower_n-tau-torch.sqrt(tu+unsquared_upper_n)

    def conf_func(tu): # Eqn 17
        tl = tl_func(tu)
        return (1-delta(n, tu)-delta(n, tl))

    tu = (lower_n - np.sqrt(unsquared_upper_n) + tau) / 2
    p = torch.tensor(tu)
    p.requires_grad_()
    optimizer = torch.optim.SGD([p], 0.0001)

    print('Before optimization (HMS): ', conf_func(p).item())
    for _ in range(100):
        optimizer.zero_grad()
        conf = conf_func(p)
       
        if conf <= 0:
            print("Negative confidence score.")
            break
        loss = 1-conf
        loss.backward()
        optimizer.step()
    print('After optimization (OMS): ', conf_func(p).item())