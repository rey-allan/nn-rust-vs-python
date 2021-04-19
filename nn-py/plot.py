import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('agg')


if __name__ == '__main__':
    with open('data/rust.txt', 'r') as f:
        rust = list(map(float, f.readlines()))
    with open('data/python.txt', 'r') as f:
        python = list(map(float, f.readlines()))

    # Plotting code referenced from:
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    labels = ['Avg', 'Max', 'Min']
    rust_scores = [round(sum(rust) / float(len(rust))), max(rust), min(rust)]
    python_scores = [round(sum(python) / float(len(python))), max(python), min(python)]

    # The locations of the labels
    x = np.arange(len(labels))
    # The width of the bars
    width = 0.35

    fig, ax = plt.subplots()
    rust_bars = ax.bar(x - width / 2, rust_scores, width, label='Rust')
    python_bars = ax.bar(x + width / 2, python_scores, width, label='Python')

    ax.set_ylabel('Runtime (milliseconds)')
    ax.set_title('Rust vs Python: Neural Network training from scratch')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', fontsize='small')

    ax.bar_label(rust_bars, padding=3)
    ax.bar_label(python_bars, padding=3)

    fig.tight_layout()

    plt.savefig('out/plot.png')
