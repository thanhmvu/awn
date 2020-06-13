import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_data(path):
    width_mults = []
    losses = []
    acc1s = []
    with open(path, 'r') as f:
        lines = f.readlines()

    for l in lines:
        words = l.split()
        if l.startswith('==> Test width'):
            width_mult = float(words[3].strip(','))
            loss = float(words[5].strip(','))
            correct, total = words[7].split('/')
            acc1 = float(correct) / float(total)

            width_mults.append(width_mult)
            losses.append(loss)
            acc1s.append(acc1)

    return (width_mults, acc1s, losses)


def save_plot(outfile, ys_list, labels, xs, title, xlabel, ylabel,
              ylim, xlim, image_size, style=',-', dpi=100):
    """
    ys_list: list of list
        list of list of data points.
        each list of data correspond to 1 curve.
    labels: list
        list of label for each curve/list of data
    """
    plt.figure()
    w, h = image_size

    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    for i in range(len(ys_list)):
        plt.plot(xs, ys_list[i], style, label=labels[i])
    plt.legend(loc="best")
    plt.savefig(outfile, dpi=dpi)
    return


def plot_acc_width(infile):
    # Extract data
    width_mults, acc1s, _ = extract_data(infile)

    # Plot
    outfile = os.path.splitext(infile)[0] + '.jpg'
    title = os.path.splitext(os.path.basename(infile))[0]
    image_size = (1000, 600)
    style = '.-'

    xlabel = 'Width Multiplier'
    xs = width_mults
    xlim = (max(xs), min(xs))

    ylabel = 'Accuracy'
    ylim = (0., 1.)
    ys_list = [acc1s]
    labels = ['accuracy']

    save_plot(outfile, ys_list, labels, xs, title, xlabel,
              ylabel, ylim, xlim, image_size, style)

