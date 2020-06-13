import sys
import shutil
import torch


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w+")
        sys.stdout = self
        return

    def write(self, message, term=True, file=True):
        if term:
            self.terminal.write(message)
        if file:
            self.log.write(message)
        return

    def flush(self):
        self.log.flush()
        return
        
    def close(self):
        sys.stdout = self.terminal
        self.log.close()
        return


def save_model(model, optimizer, epoch, train_width_mults,
               rand_width_mult_args, train_meters, val_meters, curr_val_error,
               best_val_error, filename, is_best_val, best_val_filename):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    print('Saving {} ...'.format(filename))
    model.set_width_mult(1.0)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,

        'meters': (train_meters, val_meters),
        'curr_val_error': curr_val_error,
        'best_val_error': best_val_error,
        
        'train_width_mults': train_width_mults,
        'rand_width_mult_args': rand_width_mult_args,
    }
    torch.save(state, filename)
    
    if is_best_val:
        print('Saving {} ...'.format(best_val_filename))
        shutil.copyfile(filename, best_val_filename)
    
    return


def smallest_greater(thres, list_):
    return min([x for x in list_ if x >= thres])


class NaNLossError(Exception):
    def __init__(self, value='ERROR: Loss is NaN'):
        self.value = value
    def __str__(self):
        return repr(self.value)

