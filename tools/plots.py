'''Plotting tools

2021-01-12 copied from https://github.com/jaekookang/ucm_gem_analysis
'''

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.ticker as plticker
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import interpolate
from .utils import *
sns.set(rc={'figure.facecolor': 'white'})


def make_ellipsoid(rs, r=4):
    '''Make ellipsoid data
    Note: When making CM space as 3d ellipse, (x,y,z) needs to be translated
    (x+dx,y+dy,z+dz) based on the mean of the data
    Use ax.plot_surface(x, y, z, linewidth=0, **kwargs)
    '''
    # Get rotation matrix
    U, s, rotation = np.linalg.svd(np.array(rs).T)
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    # Set cartesian coorinates
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = 0 * np.outer(np.ones_like(u), np.cos(v))  # project onto z direction!
    # Rotate
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(
                [x[i, j], y[i, j], z[i, j]], rotation)
    return x, y, z


def duration_by_condition(df, duration_col='Duration', 
                          condition_col='Vowel', condition_list=None,
                          hue=None,
                          figsize=(10,6)):
    '''Plot duration by condition (distribution)
    Args:
    - df: data frame
    - duration_col: column name for duration
    - condition_col: condition name for duration (eg., 'Vowel' column)
    - condition_list: levels for condition (eg., vowel_list=['IY1','AE1',...])
    '''
    if condition_list is None:
        condition_list = df[condition_col].unique().tolist()

    # Axes
    g = sns.displot(data=df, x=duration_col, col='Vowel', col_wrap=3,
                    height=3, kde='True', hue=hue, aspect=1.5)
    for ax, cond in zip(g.axes.flatten(), condition_list):
        median = df[duration_col].loc[df[condition_col]==cond].median()
        title_txt = f'{cond} (median={median*1000:.1f} ms)'
        ax.set_title(title_txt, fontsize=15, y=0.85, x=0.5)
        xticks = ax.get_xticks()
        xmin, xmax = xticks[0], xticks[-1]
        xticks = np.linspace(xmin, xmax, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{t*1000:.0f}' for t in xticks], fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.yaxis.get_label().set_size(20)
        ax.xaxis.get_label().set_size(20)
    # Legend
    if hue is not None:
        for txt in g.legend.get_texts():
            txt.set_size(15)
        g.legend.get_title().set_size(20)
    # Title
    plt.suptitle('Overall Vowel Duration by Rate', fontsize=25)
    # Figsize
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_facecolor('white')
    plt.tight_layout()
    return g


def duration_by_condition_boxplot(df, duration_col='Duration',
                                  condition_col='Vowel', condition_list=None,
                                  hue='Rate', hue_order=['N', 'F'],
                                  vowel2ipa=None,
                                  figsize=(10, 6), sns_context='poster'):
    '''Plot duration by condition (boxplot)
    Args:
    - df: data frame
    - duration_col: column name for duration
    - condition_col: condition name for duration (eg., 'Vowel' column)
    - condition_list: levels for condition (eg., vowel_list=['IY1','AE1',...])
    '''
    if condition_list is None:
        condition_list = df[condition_col].unique().tolist()
    # Theme
    sns.set_theme(style="whitegrid", palette="tab10", context=sns_context)
    # Figure
    fig, ax = plt.subplots(1, figsize=figsize, facecolor='white')
    # Plot
    ax = sns.violinplot(ax=ax, x=condition_col, y=duration_col,
                        hue=hue, hue_order=hue_order,
                        order=condition_list, split=True,
                        data=df)
    if isinstance(vowel2ipa, dict):
        ax.set_xticklabels([vowel2ipa[c] for c in condition_list])
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t*1000:.0f} ms' for t in yticks])
    fig.tight_layout()
    return fig, ax



def formants_by_condition(arr, df, formant_col, formant_val_col, condition_col, condition_list):
    '''Plot duration by condition
    Args:
    - arr: list of axes objects
    - df: pandas dataframe (in a long format)
    - formant_col: label for formant column; eg. 'Formant'
    - formant_val_col: label for formant value columns; eg. 'Values'
    - condition_col: label for condition column; eg. 'Word'
    - condition_list: list of condition levels; eg, ['head', 'had', ...]
    '''
    for i, (ax, level) in enumerate(zip(arr.flatten(), condition_list)):
        sns.boxplot(ax=ax, x=formant_col, y=formant_val_col,
                    data=df.loc[df[condition_col] == level])
        ax.set_title(level, fontsize=20, y=0.85, x=0.2)
        yticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
        ax.yaxis.set_ticks(yticks)
        ax.set_yticklabels([f'{t:d}' for t in yticks])
        ax.set_ylim([0, 4000])
        ax.set_ylabel('Frequencies (Hz)')
        ax.set_xlabel('Formants')
    return arr


def plot_palpha(ax, pal=None, pha=None, fontsize=None, xlim=(-70,40), ylim=(-50,30), tickspacing=20):
    if pal is not None:
        ax.plot(pal[:, 0], pal[:, 1], color='black')
    if pha is not None:
        ax.plot(pha[:, 0], pha[:, 1], color='black')
    #ax.set_xlim([-90, 50])
    #ax.set_ylim([-40, 20])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=tickspacing))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tickspacing))
    if fontsize is not None:
        ax.set_xlabel('Back <--> Front (mm)', fontsize=fontsize)
        ax.set_ylabel('Low <--> High (mm)', fontsize=fontsize)
    else:
        ax.set_xlabel('Back <--> Front (mm)')
        ax.set_ylabel('Low <--> High (mm)')
    return ax


def plot_artic(ax, data, tongue_idx, jaw_idx, lip_idx, tongue_spline=False, override_color=None, label=None, **kwargs):
    '''Plot Artic for a simple numpy array
    Args:
    - data: 2-d numpy array (single or multiple examples)
    - tongue_idx: index of the tongue sensors in `data`
    - jaw_idx: index of the jaw sensors in `data`
    - lip_idx: index of the lip sensors in `data`
    - override_color: color to override the default setting
                      This is useful to compare raw vs prediction
                      eg. ['k','r','g'] for TNG, LIP and JAW plotting colors
    2020-11-30 2-d updated
    '''
    ndim = data.shape[-1]
    data = data.reshape(-1, ndim)
    TNG = data[:,tongue_idx]
    JAW = data[:,jaw_idx]
    LIP = data[:,lip_idx]
    if override_color is None:
        colors = ['b', 'r', 'g']
    else:
        assert len(override_color) == 3, f'provide length=3 list of color codes [TNG,LIP,JAW]'
        colors = override_color
    if 'marker' not in kwargs.keys():
        marker='o'
    
    if tongue_spline:
        ax.plot(TNG[:,0::2].T, TNG[:,1::2].T,
               ls='', marker=marker, color=colors[0], zorder=1, **kwargs)
        for i in range(data.shape[0]):
            f = interpolate.interp1d(TNG[i,0::2], TNG[i,1::2], kind='quadratic')
            xnew = np.linspace(TNG[i,0], TNG[i,-2], 10, endpoint=True)
            ynew = f(xnew)
            ax.plot(xnew, ynew,
                    ls='--', marker='', color=colors[0], zorder=1, **kwargs)
    else:
        ax.plot(TNG[:,0::2].T, TNG[:,1::2].T,
               ls='', marker=marker, color=colors[0], zorder=1, **kwargs)
        ax.plot(TNG[:,0::2].T, TNG[:,1::2].T,
            ls='--', marker=marker, color=colors[0], zorder=1, **kwargs)

    ax.plot(LIP[:,0::2], LIP[:,1::2],
            ls='None', color=colors[1], marker='o', zorder=1, **kwargs)
    ax.plot(JAW[:,0::2], JAW[:,1::2],
            ls='None', color=colors[2], marker='o', zorder=1, label=label, **kwargs)
    return ax
    

def plot_artic_df(ax, df, tongue_x, tongue_y, jaw_x, jaw_y, lip_x, lip_y, 
                  tongue_spline=False, xlim=(-70,40), ylim=(-50,30), 
                  xtickspacing=20, ytickspacing=20,
                  xlabel='Horizontal (back<->front)', ylabel='Vertical (low<->high)',
                  sns_theme='whitegrid', sns_context='poster'):
    '''Plot Artic for the pandas DataFrame data
    
    *palpha should be drawn separately
    Args:
    - df: dataframe
    - tongue_x: header for tongue x; eg. ['TDx','TBx','TTx'] (back to front)
    - tongue_y: header for tongue y; eg. ['TDy','TBy','TTy'] (back to front)
    - jaw_x: header for jaw x; eg. ['JAWx']
    - jaw_y: header for jaw y; eg. ['JAWy']
    - lip_x: header for lip x; eg. ['ULx','LLx']
    - lip_y: header for lip y; eg. ['ULy','LLy']
    - (optional) tongue_spline: if true, draw spline on the tongue sensors only
    '''
    TNG_x = df[tongue_x]
    TNG_y = df[tongue_y]
    JAW_x = df[jaw_x]
    JAW_y = df[jaw_y]
    LIP_x = df[lip_x]
    LIP_y = df[lip_y]

    # Theme
    sns.set_theme(style=sns_theme, context=sns_context)
    # Spline
    if not tongue_spline:  # no spline
        ax.plot(TNG_x, TNG_y, ls='None', marker='o', markersize=3, zorder=1)
    else:  # yes spline
        for xs, ys in zip(TNG_x.values, TNG_y.values):
            f = interpolate.interp1d(xs, ys, kind='slinear')
            xnew = np.linspace(xs[0], xs[-1], 10, endpoint=True)
            ynew = f(xnew)
            ax.plot(xs, ys,ls='', marker='o', color='b', markersize=3, zorder=1)
            ax.plot(xnew, ynew, ls='--', lw=0.5, color='gray', marker='', alpha=0.8, zorder=1)
    # Lips
    ax.plot(LIP_x, LIP_y, ls='None', color='r', marker='o', markersize=3, zorder=1)
    # Jaw
    ax.plot(JAW_x, JAW_y, ls='None', color='gray', marker='o', markersize=3, zorder=1)
    # Prettify
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=xtickspacing))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=ytickspacing))
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.gcf().tight_layout()
    return ax


def plot_F1F2_ref(ax, F1, F2, labels, textcolor='gray', use_mel=False, alpha=0.5):
    # Plot F1-F2 reference vowels
    # F1, F2: column vector
    ax.plot(F2, F1, ls='None', marker='o', markersize=5, color=textcolor, alpha=alpha)
    for i, v in enumerate(labels):
        ax.text(F2[i], F1[i], v, color=textcolor, fontsize=20, alpha=alpha)
    if use_mel:
        ax.set_xlim([2000, 800])
        ax.set_ylim([1200, 200])
        suffix = ' (mel)'
    else:
        ax.set_xlim([3000, 400])
        ax.set_ylim([1000, 100])
        suffix = ''
    ax.set_xlabel('F2' + suffix)
    ax.set_ylabel('F1' + suffix)
    return ax


def plot_F2F3_ref(ax, F2, F3, labels, use_mel=False):
    # Plot F2-F3 reference vowels
    # F2, F3: column vector
    ax.plot(F2, F3, ls='None', marker='*', markersize=5, color='gray')
    if use_mel:
        ax.set_xlim([2000, 800])
        ax.set_ylim([2500, 1200])
        suffix = ' (mel)'
    else:
        ax.set_xlim([3000, 400])
        ax.set_ylim([4000, 1500])    
        suffix = ''
    ax.set_xlabel('F2' + suffix)
    ax.set_ylabel('F3' + suffix)
    for i, v in enumerate(labels):
        ax.text(F2[i], F3[i], v, color='gray')
    return ax


def plot_F1F2(ax, F1, F2, show_label=True, use_mel=False, show_title=False, tickspacing=200, **kwargs):
    # F2-F1
    ax.grid(True)
    if show_title:
        ax.set_title('F2-F1', fontsize=15)
    # ax1.scatter(F2, F1, s=5, color='b')
    ax.plot(F2, F1, 'o', **kwargs)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=tickspacing))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=tickspacing))
    if use_mel:
        ax.set_xlim([2000, 800])
        ax.set_ylim([1200, 200])
        suffix = ' (mel)'
    else:
        ax.set_xlim([3000, 400])
        ax.set_ylim([1000, 100])
        suffix = ''
    ax.set_xlabel('F2' + suffix)
    ax.set_ylabel('F1' + suffix)
    if show_label & isinstance(F2, float):
        ax.set_xlabel(f'F2={F2:.1f},  F1={F1:.1f}')
    return ax


def plot_F2F3(ax, F2, F3, show_label=True, use_mel=False, **kwargs):
    # F3-F2
    ax.grid(True)
    ax.set_title('F2-F3', fontsize=20)
    # ax2.scatter(F2, F3, s=5, color='b')
    ax.plot(F2, F3, 'o', **kwargs)
    ax.invert_xaxis()
    ax.invert_yaxis()
    if use_mel:
        ax.set_xlim([2000, 800])
        ax.set_ylim([2500, 1200])
        suffix = ' (mel)'
    else:
        ax.set_xlim([3000, 400])
        ax.set_ylim([4000, 1500])    
        suffix = ''
    ax.set_xlabel('F2' + suffix)
    ax.set_ylabel('F3' + suffix)
    if show_label & isinstance(F2, float):
        ax.set_xlabel(f'F3={F3:.1f},  F2={F2:.1f}')
    return ax


def plot_F1F2F3(axes, F1, F2, F3, show_label=True, use_mel=False, **kwargs):
    ax1, ax2 = axes
    if use_mel:
        F1_range = [1200, 200]
        F2_range = [2000, 800]
        F3_range = [2500, 1200]
        suffix = ' (mel)'
    else:
        F1_range = [1000, 100]
        F2_range = [3000, 400]
        F3_range = [4000, 1500]
        suffix = ''

    # F2-F1
    ax1.grid(True)
    ax1.set_title('F2-F1', fontsize=20)
    # ax1.scatter(F2, F1, s=5, color='b')
    ax1.plot(F2, F1, 'o', **kwargs)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_xlim(F2_range)
    ax1.set_ylim(F1_range)
    ax1.set_xlabel('F2' + suffix)
    ax1.set_ylabel('F1' + suffix)
    if show_label & isinstance(F2, float):
        ax1.set_xlabel(f'F2={F2:.1f},  F1={F1:.1f}' + suffix)

    # F3-F2
    ax2.grid(True)
    ax2.set_title('F2-F3', fontsize=20)
    # ax2.scatter(F2, F3, s=5, color='b')
    ax2.plot(F2, F3, 'o', **kwargs)
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    ax2.set_xlim(F2_range)
    ax2.set_ylim(F3_range)
    ax2.set_xlabel('F2' + suffix)
    ax2.set_ylabel('F3' + suffix)
    if show_label & isinstance(F2, float):
        ax2.set_xlabel(f'F3={F3:.1f},  F2={F2:.1f}' + suffix)
    return ax1, ax2

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', return_params=False, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    2020-08-30 Jaekoo edited (See https://github.com/jaekookang/ucm_gem_analysis/blob/master/procs/IEEE/04-4_plot_vowel_ellipse.ipynb)
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    width = ell_radius_x * 2
    height = ell_radius_y * 2
    ellipse = Ellipse((0, 0),
        width=width,
        height=height,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_y, scale_x) \
        .translate(mean_y, mean_x)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    
    if not return_params:
        return ax, (scale_x/n_std, scale_y/n_std)
    else:
        param = {
            'ellipse': ellipse,
            'transform': transf,
            'width': width,
            'height': height,
            'scale_y': scale_y,
            'scale_x': scale_x,
            'mean_y': mean_y,
            'mean_x': mean_x,
            'angle': 45
        }
        return ax, (scale_x/n_std, scale_y/n_std), param