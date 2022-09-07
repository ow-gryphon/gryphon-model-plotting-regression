import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.reset_orig()
from . import diagnostics_continuous


def scatter_plot_comparison(frame, actual, predicted, figsize=(10,6), N = 1000, equalize_axes=True, colormap = None):

    if isinstance(predicted, str):
        predicted = [predicted]
        
    fig, ax = plt.subplots(figsize=figsize)

    if frame.shape[0] > N:
        sampled_frame = frame.sample(N)
        print("Sampled {} out of {} observations".format(N, frame.shape[0]))
    else:
        sampled_frame = frame
    
    if colormap is None:
        colors = cm.rainbow(np.linspace(0, 1, len(predicted)))
    else:
        colors = iter(colormap(np.linspace(0, 1, len(predicted))))
    
    for predicted_var, color in zip(predicted, colors):
        sampled_frame.plot(kind='scatter', y=actual, x=predicted_var, ax = ax, color=[color], alpha = 0.3)

    # Format the axis
    min_x = ax.get_xlim()[0]
    min_y = ax.get_ylim()[0]
    max_x = ax.get_xlim()[1]
    max_y = ax.get_ylim()[1]
    
    min_axis = min(min_x, min_y)
    max_axis = max(max_x, max_y)
        
    # Add a 45 degree line
    ax.plot([(min_axis, min_axis),(max_axis, max_axis)],
            [(min_axis, min_axis),(max_axis, max_axis)], color='black', linewidth = 0.5)
    
    if equalize_axes:
        ax.set_xlim(min_axis, max_axis)
        ax.set_ylim(min_axis, max_axis)
    else:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    
    # Label the axes
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
        
    # Add legend
    plt.legend(predicted, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11);

    return fig, ax
    
    
def act_vs_pred_comparison(frame, actual, predicted, figsize=(10,6), bins=10, equalize_axes=True, colormap = None):

    if isinstance(predicted, str):
        predicted = [predicted]
    
    tables = []
    fig, ax = plt.subplots(figsize=figsize)
    
    if colormap is None:
        colors = cm.rainbow(np.linspace(0, 1, len(predicted)))
    else:
        colors = iter(colormap(np.linspace(0, 1, len(predicted))))
    
    for x_var, color in zip(predicted, colors):
        var_table = diagnostics_continuous.act_vs_pred_plot(used_data = frame, 
                                            actual_var = actual, 
                                            pred_var = x_var, 
                                            num_buckets=bins,
                                            with_count=False,
                                            no_plot=True)
        
        var_table.plot(kind='scatter', y="Actual", x=x_var, ax = ax, color=[color], alpha = 0.7)
        
        var_table['Var'] = x_var
        var_table=var_table.rename(columns={x_var: "Pred"})
    
        tables.append(var_table)
        
    table_to_plot = pd.concat(tables)
    
    # Format the axis
    min_x = ax.get_xlim()[0]
    min_y = ax.get_ylim()[0]
    max_x = ax.get_xlim()[1]
    max_y = ax.get_ylim()[1]
    
    min_axis = min(min_x, min_y)
    max_axis = max(max_x, max_y)
        
    # Add a 45 degree line
    ax.plot([(min_axis, min_axis),(max_axis, max_axis)],
            [(min_axis, min_axis),(max_axis, max_axis)], color='black', linewidth = 0.5)
    
    if equalize_axes:
        ax.set_xlim(min_axis, max_axis)
        ax.set_ylim(min_axis, max_axis)
    else:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    
    # Label the axes
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
        
    # Add legend
    plt.legend(predicted, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=11);

    return fig, ax, table_to_plot