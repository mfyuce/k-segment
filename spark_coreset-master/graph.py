#!/usr/bin/env python
__author__ = 'Ahmad Yasin'
import matplotlib.pyplot as plt
import numpy as np
import sys, argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='results.txt', help='input txt file')
    parser.add_argument('-v', '--verbose', default=0, help='verbose printing?')
    args = parser.parse_args()
    return args
 
def plot(x_ax, cs, s1, s2=None, s3=None, show=True, labels=None):
    y_cs = cs[0]; y_sp2 = s1[0]
    t_cs = cs[1]; t_csg = cs[2]; t_sp = s1[1]
    if labels is None: labels = ['no-x-label', 'no-title-1', 'no-title-2']
    
    plt.figure(1)
    plt.subplot(211)
    ln_cor, = plt.plot(x_ax, y_cs, 'r', marker='.', label="coreset")
    if y_sp2 is not None:
        ln_sp2, = plt.plot(x_ax, y_sp2, 'g', label="sp-km||", marker='p')
    if s2 is not None:
        ln_sp1, = plt.plot(x_ax, s2[0], 'y', marker='x', label="sp-rand")
    if s3 is not None:
        ln_uni, = plt.plot(x_ax, s3[0], 'b', marker='*', label="uniform")
    #    plt.legend([ln_cor, ln_uni, ln_sp1, ln_sp2], ['coreset', 'uniform', 'sp-rand', 'sp-kmpp'], loc='best')
    #else:
    #    plt.legend([ln_cor, ln_sp2], ['coreset', 'sp-kmpp'], loc='best')
    plt.ylabel('Relative cost Avg error' if y_sp2 is not None else 'cost over spark')
    #plt.gcf().savefig('figure_cost.png')
    plt.title(labels[1] + '\n' + labels[2]);
           
    plt.subplot(212) # Time plot starts here
    ln_cs, = plt.plot(x_ax, t_cs, 'r', marker='.', label="coreset")
    ln_sp, = plt.plot(x_ax, t_sp, 'g', marker='p', label="sp-km||")
    lines = [ln_sp, ln_cs]
    titles = ['sp-km||', 'coreset']
    if cs[2] is not None:
        ln_csg, = plt.plot(x_ax, t_csg, 'r', marker='.', label="cset-gen", ls='--')
        lines.append(ln_csg)
        titles.append('cset-gen')        
    if s2:
        lines.append(ln_sp1)
        ln_sp1 = plt.plot(x_ax, s2[1], 'y', marker='x', label="sp-rand")
        titles.append('sp-rand')
    if s3 is not None and s3[1] is not None:
        ln_un, = plt.plot(x_ax, s3[1], 'b', marker='*', label="uniform")
        lines.append(ln_un)
        titles.append('uniform')
    plt.legend(lines, titles, loc='best')
    #plt.legend([ln_cs, ln_sp], ['coreset', 'sp-kmpp'], loc='best')
    #plt.title(runtag1); plt.xlabel(xlabel); plt.ylabel('Avg Time [s]')
    plt.ylabel('Avg Time [s]')
    #if plot[2]>1:   plt.show()
    plt.xlabel(labels[0]); 
    plt.gcf().savefig('figure_both.png')
    if show:   plt.show()

def main(argv):
    args = get_args()
    v = int(args.verbose)
    d = np.loadtxt(args.input)#, skiprows=1)
    t = np.loadtxt(args.input + '1', dtype=str, delimiter='@')
    if v: print 't=', t
    y1 = d[:,1]
    y2 = d[:,2]
    if v>1:
        y1 /= y2
        y2 = None
    
    plot(d[:,0], 
        [ y1, d[:,3], d[:,5] ],
        [ y2, d[:,4] ],
        labels=[ t[0] + '  ' + t[3], t[1], t[2] ] );
    
if __name__ == "__main__":
    main(sys.argv)

