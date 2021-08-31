'''Post-processing Tools for Haskins IEEE data

2021-01-12 copied from https://github.com/jaekookang/ucm_gem_analysis/blob/master/tools/postproc_ieee.py
'''

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib.ticker as plticker
import re
import pickle
import numpy as np
import pandas as pd
from .utils import find_elements
from .plots import confidence_ellipse


def remove_short_tokens(df, which_spkr, spkr_col='Subj', dur_col='Dur', lower_threshold=0.03, upper_threshold=1.):
    '''Remove short tokens (default < 30ms)
    Args:
    - df: pandas data frame
    - which_spkr: speaker id under speaker column in df
    - spkr_col: name of the speaker column in df
    - dur_col: name of the duration column in df
    - lower_threshold: lower cutoff duration threshold (sec); default=0.03s
    - upper_threshold: upper cutoff duration threshold (sec); default=1s
    Returns:
    - df: modified df
    '''
    # Copy first to prevent overwritting
    df = df.copy()

    n_samples = df.loc[df[spkr_col] == which_spkr].shape[0]
    tmp = df.loc[(df[spkr_col] == which_spkr)]
    tmp = tmp.loc[(tmp[dur_col] < lower_threshold) |
                  (tmp[dur_col] > upper_threshold)]
    n_trimmed = tmp.shape[0]
    df.drop(tmp.index, inplace=True)
    print(f'{which_spkr}: {n_samples} --> {n_samples-n_trimmed} ({n_trimmed})')
    return df


def iqr_bounds(df, col, lb=0.25, ub=0.75):
    '''
    IQR covers (ub-lb)*100 % of data from the median
      lb: lower bound (default=0.25)
      ub: upper bound (default=0.75)
    '''
    q1 = df[col].quantile(lb)
    q3 = df[col].quantile(ub)
    IQR = q3 - q1
    lrbound = q1 - 1.5*IQR
    upbound = q3 + 1.5*IQR
    return lrbound, upbound


def remove_acous_outlier(df, which_spkr, which_vowel, spkr_col='Speaker', label_col='Vowel', lower_bound=0.25, upper_bound=0.75):
    '''Remove outliers in acoustics data (F1, F2, F3)
    lb: lower bound
    ub: upper bound
    '''
    df = df.copy()
    d = df.loc[(df[spkr_col] == which_spkr) & (df[label_col] == which_vowel)]

    # F1
    lb, ub = iqr_bounds(d, 'F1', lb=lower_bound, ub=upper_bound)
    f1outLier = df.loc[(df[spkr_col] == which_spkr) & (
        df[label_col] == which_vowel) & ((df.F1 < lb) | (df.F1 > ub))]
    df.drop(f1outLier.index, inplace=True)  # no need to reassign
    # F2
    lb, ub = iqr_bounds(d, 'F2', lb=lower_bound, ub=upper_bound)
    f2outLier = df.loc[(df[spkr_col] == which_spkr) & (
        df[label_col] == which_vowel) & ((df.F2 < lb) | (df.F2 > ub))]
    df.drop(f2outLier.index, inplace=True)  # no need to reassign
    # F3
    lb, ub = iqr_bounds(d, 'F3', lb=lower_bound, ub=upper_bound)
    f3outLier = df.loc[(df[spkr_col] == which_spkr) & (
        df[label_col] == which_vowel) & ((df.F3 < lb) | (df.F3 > ub))]
    df.drop(f3outLier.index, inplace=True)  # no need to reassign
    return df, (f1outLier, f2outLier, f3outLier)


def load_palate(palate_files, spkr_list):
    # Load palate files into per-speaker dictionary
    # eg. {spkr: pal}
    pal_all = {}
    for spkr in spkr_list:
        _, pal_file = find_elements(spkr, palate_files)
        with open(pal_file[0], 'rb') as pckl:
            pal = pickle.load(pckl)[:, [0, 2]]
        pal_all.update({spkr: pal})
    return pal_all


def to_long(dataframe):
    '''Convert wide format to long format for Formants
    Note: "suffix" parameter must be specified!
    '''
    dfw = dataframe.copy()
    dfw['id'] = dfw.index
    df = pd.wide_to_long(dfw, stubnames='F', sep='', suffix='\\d+',
                         i=['FileID', 'SpkrID', 'Sex', 'Block', 'Rate', 'Time',
                             'Word', 'Phone', 'PhoneDur', 'WordDur', 'SentDur', 'id'],
                         j='Type').reset_index()
    df.rename(columns={'Type': 'Formant', 'F': 'Values'}, inplace=True)

    df.Formant = df.Formant.astype('category')
    df.Formant.cat.rename_categories({1: 'F1', 2: 'F2', 3: 'F3'}, inplace=True)
    return df


def plot_duration(ax, df, which_spkr, vowels, vowel2ipa,
                  sns_style='whitegrid',
                  sns_context='poster',
                  palette="Set2"):
    df = df.loc[(df.Speaker == which_spkr) & (df.Vowel.isin(vowels))]
    # Theme
    sns.set_theme(style=sns_style, context=sns_context, palette=palette)
    # Plot
    sns.boxplot(ax=ax, x='Vowel', y='Duration', order=vowels,
                hue='Rate', hue_order=['N', 'F'],
                data=df)
    # Style
    ax.set_xlabel('Vowels', fontsize=20)
    ax.set_ylabel('Duration (ms)', fontsize=20)
    ax.xaxis.set_ticklabels([vowel2ipa[v] for v in vowels])
    yticks = np.arange(0, 6)/10
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t*1000:.0f}' for t in yticks])
    ax.legend(loc=1)
    ax.get_legend().texts[0].set_text('Normal')
    ax.get_legend().texts[1].set_text('Fast')
    ax.get_legend().texts[0].set_fontsize(20)
    ax.get_legend().texts[1].set_fontsize(20)
    ax.set_title('Duration', fontsize=40, y=1.02)
    return ax


def plot_F1F2_df(ax, df, vowel_col='Vowel', vowel_list=None, colors=None,
                 F1_col='F1_mel', F2_col='F2_mel',
                 F1_range=[200, 1200], F2_range=[800, 2000],
                 return_colors=False,
                 vowel2ipa=None,
                 show_label=True,
                 ellipse_linestyle='--',
                 ellipse_only=False,
                 center_only=False,
                 connected_linestyle='-',
                 sns_theme='whitegrid',
                 sns_context='poster',
                 palette='tab10',
                 return_params=False):
    '''Plot F1-F2 using dataframe (default: mel scale)
    which_rate: 'N' or 'F' (IEEE dataset)
    vowel_col: vowel column name (eg., 'Vowel')
    vowel_list: vowel list (eg., ['IY1','AE1',...]) as hue_order in the plot
                make sure they are all possible unique levels in vowel_col
    colors: color list
    hue: hue column name (eg., 'Rate'). See sns.scatterplot
    hue: hue order. See sns.scatterplot
    F1_col: F1 column name
    F2_col: F2 column name
    show_label: if True (default), show the vowel labels
    ellipse_only: if False (default), it will plot individual data points with ellipse
                  if True, it will only plot ellipses
    center_only: if False (defalt), it will show entire point distributions,
                 if specified either 'mean' or 'median', it will only show that parameter with connected lines

    '''
    front_vowels = ["IY1", "IH1", "EH1", "AE1"]
    back_vowels = ["AH1", "AA1", "AO1", "UH1", "UW1"]

    if vowel_list is None:
        vowel_list = df[vowel_col].unique().tolist()
        assert len(
            vowel_list) == 0, f'No vowels found. Provide valid vowel list/levels'
    # else:
    #     df = df.loc[df[vowel_col].isin(vowel_list)].copy()
    if colors is None:
        colors = sns.color_palette('tab10', len(vowel_list))
    else:
        assert len(vowel_list) == len(
            colors), 'vowel_list and colors do not match'
    # Adjust df
    df = df.loc[df[vowel_col].isin(vowel_list)]

    # Theme
    sns.set_theme(style=sns_theme, context=sns_context, palette=palette)
    # Plot data
    if (center_only is False) & (ellipse_only is False):
        sns.scatterplot(ax=ax, x=F2_col, y=F1_col, hue=vowel_col, s=5,
                        palette=colors, hue_order=vowel_list,
                        data=df)

    ax.set_xlim(F2_range)
    ax.set_ylim(F1_range)
    ax.invert_xaxis()
    ax.invert_yaxis()
    # Add ellipse
    F1s = []
    F2s = []
    params = {v:'' for v in vowel_list}
    for which_vowel, color in zip(vowel_list, colors):
        F1 = df[F1_col].loc[df[vowel_col] == which_vowel].values
        F2 = df[F2_col].loc[df[vowel_col] == which_vowel].values

        if center_only is False:
            # Plotting the distribution
            ax, (x_std, y_std), p = confidence_ellipse(F1, F2, ax, 
                                                            n_std=1.96,
                                                            facecolor='none',
                                                            edgecolor=color, 
                                                            linestyle=ellipse_linestyle,
                                                            return_params=True)
            params[which_vowel] = p

        elif center_only == 'mean':
            # Plotting vowel-centers and connected to make ovoids
            mF1, mF2 = F1.mean(), F2.mean()
            ax.plot(mF2, mF1, 'ko', markersize=3)
            F1s += [mF1]
            F2s += [mF2]
        elif center_only == 'median':
            mF1, mF2 = np.median(F1), np.median(F2)
            ax.plot(mF2, mF1, 'ko', markersize=3)
            F1s += [mF1]
            F2s += [mF2]
        else:
            raise Exception(
                f'Provide either "mean", "median", or None, not {center_only}')

        # Add vowel labels
        if which_vowel in front_vowels:
            f2_shift = 60
        if which_vowel in back_vowels:
            f2_shift = -40

        if show_label:
            if vowel2ipa is not None:
                txt = ax.text(np.mean(F2)+f2_shift, np.mean(F1)+20,
                              vowel2ipa[which_vowel], fontsize=20, color='k')
            else:
                txt = ax.text(np.mean(F2)+f2_shift, np.mean(F1)+20,
                              which_vowel, fontsize=20, color='k')
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=5, foreground='w')])

    # Add center lines (only when drawing ovoids)
    if center_only in ['mean', 'median']:
        ax.plot(F2s+[F2s[0]], F1s+[F1s[0]], 'k',
                linestyle=connected_linestyle, linewidth=1)

    # Turn off legend
    if (center_only is False) & (ellipse_only is False):
        ax.get_legend().remove()
    # Axis
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    if re.search('mel', F1_col):
        ax.set_xlabel('F2 (mel)', fontsize=20)
        ax.set_ylabel('F1 (mel)', fontsize=20)
    else:
        ax.set_xlabel('F2', fontsize=20)
        ax.set_ylabel('F1', fontsize=20)
    # tighten
    plt.gcf().tight_layout()
    
    returned = [ax]
    if return_colors:
        return returned + [colors]
    if return_params:
        return returned + [params]
    else:
        return returned[0]


def plot_F2F3_df():
    raise NotImplementedError


def _duration_by_condition(df, duration_col='Duration',
                           condition_col='Vowel', condition_list=None,
                           hue=None, figsize=(10, 6)):
    '''Plot duration by condition
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
        median = df[duration_col].loc[df[condition_col] == cond].median()
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


def plot_dur_formants_by_rate(df, arr, vowel_list, title='Plot by Speech Rate',
                               x='Phone', y='PhoneDur', hue='Rate', use_mel=False):
    '''Plot duration and formant distribution by speech rate
    2020-09-28: this function is not longer used (not pretty)
                instead, use different plot modules and combine as a single figure
    '''
    ax1, ax2, ax3, ax4, ax5 = arr
    marker_size = 2
    if use_mel:
        F1_range = [1200, 200]
        F2_range = [2000, 500]
        F3_range = [2500, 1200]
        F1_label = 'F1 (mel)'
        F2_label = 'F2 (mel)'
        F3_label = 'F3 (mel)'
    else:
        F1_range = [1250, 100]
        F2_range = [2200, 500]
        F3_range = [4000, 1500]
        F1_label = 'F1'
        F2_label = 'F2'
        F3_label = 'F3'
    sns.boxplot(x=x, y=y, order=vowel_list,
                hue=hue, hue_order=['N', 'F'],
                data=df, palette="Set3", ax=ax1)
    yheight = 0.4
    ax1.set_yticks(np.linspace(0, yheight, 6))
    ax1.set_yticklabels([f'{int(t*1000):d}' for t in ax1.get_yticks()])
    ax1.set_ylabel('Duration (ms)')
    ax1.set_ylim([0, yheight])

    for v in vowel_list:
        # ----- Normal rate
        d = df.loc[(df[x] == v) & (df[hue] == 'N')]
        if use_mel:
            F1, F2, F3 = d.F1_mel, d.F2_mel, d.F3_mel
        else:
            F1, F2, F3 = d.F1, d.F2, d.F3
        F1_md, F2_md, F3_md = F1.median(), F2.median(), F3.median()

        ax2.set_title('Normal rate (F2-F1)')
        ax2.scatter(F2, F1, s=marker_size)
        ax2.set_xlim(F2_range)
        ax2.set_ylim(F1_range)
        ax2.set_xlabel(F2_label)
        ax2.set_ylabel(F1_label)
        ax2.text(F2_md, F1_md, v, fontsize=10, fontweight='bold')

        ax3.set_title('Normal rate (F2-F3)')
        ax3.scatter(F2, F3, s=marker_size)
        ax3.set_xlim(F2_range)
        ax3.set_ylim(F3_range)
        ax3.set_xlabel(F2_label)
        ax3.set_ylabel(F3_label)
        ax3.text(F2_md, F3_md, v, fontsize=10, fontweight='bold')

        # ----- Fast rate
        d = df.loc[(df[x] == v) & (df[hue] == 'F')]
        if use_mel:
            F1, F2, F3 = d.F1_mel, d.F2_mel, d.F3_mel
        else:
            F1, F2, F3 = d.F1, d.F2, d.F3
        F1_md, F2_md, F3_md = F1.median(), F2.median(), F3.median()
        ax4.set_title('Fast rate (F2-F1)')
        ax4.scatter(F2, F1, s=marker_size)
        ax4.set_xlim(F2_range)
        ax4.set_ylim(F1_range)
        ax4.set_xlabel(F2_label)
        ax4.set_ylabel(F1_label)
        ax4.text(F2_md, F1_md, v, fontsize=10, fontweight='bold')

        ax5.set_title('Fast rate (F2-F3)')
        ax5.scatter(F2, F3, s=marker_size)
        ax5.set_xlim(F2_range)
        ax5.set_ylim(F3_range)
        ax5.set_xlabel(F2_label)
        ax5.set_ylabel(F3_label)
        ax5.text(F2_md, F3_md, v, fontsize=10, fontweight='bold')

    ax3.get_shared_x_axes().join(ax3, ax5)
    ax3.get_shared_y_axes().join(ax3, ax5)
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.05)
    return arr
