import re
import os
import csv
import argparse
from collections import defaultdict


def parse_profile(path):
    blocks = {}
    cur_models = None
    pattern_file = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+?):\d+\(([^)]+)\)\s*$')
    pattern_builtin = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\{.+\})\s*$')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            m = re.match(r'^##\s*(\d+)', line)
            if m:
                cur_models = int(m.group(1))
                blocks[cur_models] = {}
                continue
            if cur_models is None:
                continue
            m1 = pattern_file.match(line)
            if m1:
                cumtime = float(m1.group(4))
                filename = m1.group(6)
                funcname = m1.group(7)
                key = f"{os.path.basename(filename)}:{funcname}"
                blocks[cur_models][key] = cumtime
                continue
            m2 = pattern_builtin.match(line)
            if m2:
                cumtime = float(m2.group(4))
                functext = m2.group(6)
                key = functext.strip('{} ').strip()
                blocks[cur_models][key] = cumtime
    return blocks


def write_pivot_csv(blocks, outpath):
    models_sorted = sorted(blocks.keys())
    all_funcs = sorted({fn for b in blocks.values() for fn in b.keys()})
    with open(outpath, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['models'] + all_funcs)
        for m in models_sorted:
            row = [m] + [blocks[m].get(fn, '') for fn in all_funcs]
            writer.writerow(row)


def plot_top_functions(blocks, outpath, top_n=8):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError('plotly is required to create interactive plots') from e

    models_sorted = sorted(blocks.keys())
    all_funcs = sorted({fn for b in blocks.values() for fn in b.keys()})
    # choose top_n functions by maximum cumtime
    func_max = sorted(all_funcs, key=lambda f: max(blocks[m].get(f, 0) for m in models_sorted), reverse=True)[:top_n]

    fig = go.Figure()
    for fn in func_max:
        y = [blocks[m].get(fn, 0) for m in models_sorted]
        fig.add_trace(go.Scatter(x=models_sorted, y=y, mode='lines+markers', name=fn,
                                 hovertemplate='%{y:.3f}s<br>models=%{x}<extra>' + fn + '</extra>'))

    fig.update_layout(title='Profile scaling vs number of models',
                      xaxis_title='number of models',
                      yaxis_title='cumtime (s)',
                      template='plotly_white')

    # write interactive HTML
    fig.write_html(outpath, include_plotlyjs='cdn')
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Parse profiling blocks and plot scaling')
    parser.add_argument('infile', nargs='?', default='prototyping/lci_scaling_time.txt')
    parser.add_argument('--csv', default='prototyping/lci_scaling_times.csv')
    parser.add_argument('--plot', default='prototyping/lci_scaling_plot.html')
    parser.add_argument('--top', type=int, default=8, help='number of top functions to plot')
    args = parser.parse_args()

    blocks = parse_profile(args.infile)
    if not blocks:
        print('no blocks found in', args.infile)
        return

    write_pivot_csv(blocks, args.csv)
    print('wrote CSV to', args.csv)
    try:
        plot_top_functions(blocks, args.plot, top_n=args.top)
        print('wrote plot to', args.plot)
    except RuntimeError as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
