"""
Source: https://github.com/JiahuiYu/slimmable_networks
"""
import time
from decimal import Decimal


class Meter(object):
    """ Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.

    Args:
        name (str): the name of meter

    """
    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.reset()

    def reset(self):
        self.values = []
        
    def size(self):
        return len(self.values)

    def cache(self, value, pstep=1):
        self.steps += pstep
        self.values.append(value)

    def cache_list(self, value_list, pstep=1):
        self.steps += pstep
        self.values += value_list

    def flush(self, value, reset=True):
        pass


class ScalarMeter(Meter):
    """ ScalarMeter records scalar over steps. """
    def __init__(self, name):
        super(ScalarMeter, self).__init__(name)

    def flush(self, value, step=-1, reset=True):
        if reset:
            self.reset()


def flush_scalar_meters(meters, method='avg', flush=True):
    """ Docstring for flush_scalar_meters """
    results = {}
    assert isinstance(meters, dict), "meters should be a dict."
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            continue
        if method == 'avg':
            n = len(meter.values)
            value = -1 if n==0 else sum(meter.values) / n
        elif method == 'sum':
            value = sum(meter.values)
        elif method == 'max':
            value = max(meter.values)
        elif method == 'min':
            value = min(meter.values)
        else:
            raise NotImplementedError(
                'flush method: {} is not yet implemented.'.format(method))
        results[name] = value
        if flush:
            meter.flush(value)
    return results


def flush_wm_meters(meters, method='avg', flush=True):
    output = {}
    for key in meters:
        try:
            float(key)
            width_mult = key
        except ValueError:
            continue

        results = flush_scalar_meters(meters[width_mult], method, flush)
        err1 = results['top1_error']
        err5 = results['top5_error']

        output[width_mult] = {}
        output[width_mult]['loss'] = results['loss']
        output[width_mult]['err1'] = err1
        output[width_mult]['err5'] = err5
        output[width_mult]['acc1'] = -1 if (err1 == -1) else (1 - err1)
        output[width_mult]['acc5'] = -1 if (err5 == -1) else (1 - err5)
    return output
    
