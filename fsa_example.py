import numpy as np
from fsa_util import slack_oms



if __name__ == '__main__':
    print('SNLI exp:')
    slack_oms(tau=0.0, lower_n=0.899, unsquared_upper_n=0.879**2, n=10000)
    print()
    slack_oms(tau=0.0, lower_n=0.919, unsquared_upper_n=0.879**2, n=10000)
    print()

    print('SST-2 exp:')
    slack_oms(tau=0.0, lower_n=0.949, unsquared_upper_n=0.939**2, n=1821)
    print()
    slack_oms(tau=0.0, lower_n=0.971, unsquared_upper_n=0.939**2, n=1821)
    print()

