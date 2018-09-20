from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time


def get_time():
    return time.time()

def convert_time(t):
    hours = np.floor(t/3600.0)
    minutes = np.floor((t/3600.0 - hours) * 60)
    seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
    if hours > 0:
        minutes = np.floor((t/3600.0 - hours) * 60)
        seconds = np.ceil(((t/3600.0 - hours) * 60 - minutes) * 60)
        t_str = str(int(hours)) + 'h  ' + \
                str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    elif (hours == 0) and (minutes >= 1):
        minutes = np.floor(t/60.0)
        seconds = np.ceil((t/60.0 - minutes) * 60)
        t_str = str(int(minutes)).rjust(2) + 'm  ' + \
                str(int(seconds)).rjust(2) + 's'
    else:
        seconds = (t/60.0 - minutes) * 60
        t_str = str(seconds) + 's'

        
    return t_str


