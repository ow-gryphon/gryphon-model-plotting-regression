import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from . import exploration_utilities
import seaborn as sns
sns.reset_orig()
import scipy.stats as ss


# Internal functions used to transform data
def _linear(x):
    return x
def _log(x):
    return np.log(x)
def _logit(x):
    return np.log(x/(1-x))
def _exp(x):
    return np.exp(x)
def _invlogit(x):
    return 1/(1+exp(-x))

def _log1p_abs(x):
    return np.sign(x)*np.log10(1+abs(x))


_scaling_functions = {"linear": _linear, "log": _log, "logit": _logit}
_inverse_scaling_functions = {"linear": _linear, "log": _exp, "logit": _invlogit}


def bivariate_continuous(used_data, y_var, x_var, num_buckets=20, y_scale="linear", x_scale="linear",
                         with_count=True, with_stderr=False, with_CI = False,
                         lower=-np.inf, upper=np.inf, trendline=False, header=None):
    """
    Generates bivariate plots for continuous variables, where the data is bucketed based on x-variable percentiles.
    Provides option to scale the y-axis and count of observations in each bucket (in case there is concentration)
    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param y_var: a single string with the name of the y-variable
    :param x_var: a single string with the name of the x-variable
    :param num_buckets: Integer number of buckets (quantiles) for which to group the x-variable values
    :param y_scale: Scale the y-axis, either: 'linear', 'log', 'logit', or 'symlog'
    :param x_scale: Scale the x-axis, either: 'linear', 'log', 'logit', or 'symlog'
    :param with_count: Boolean indicating whether bars representing number of observations should be
    plotted on the secondary y-axis
    :param with_stderr: Boolean indicating whether standard error bars are added to the plot
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    (Note that this is only valid for dependent variables that are 0 or 1, and will override stderr if used)
    :param lower:
        For with_stderr: Double indicating the lowest value for the error bars (e.g. 0 for binary data)
        For with_CI: Lower probability
    :param upper:
        For with_stderr: Double indicating the highest value for the error bars (e.g. 1 for binary data)
        For with_CI: Upper probability
    :param trendline: Boolean whether to include a linear trendline. Not possible with the 'symlog' scalar
    :param header: Optional string for the main title of the plot
    :return: Tuple of Figure and Pandas dataframe containing data for the figure
    """

    used_data = used_data.assign(x_bin=pd.qcut(used_data[x_var], int(num_buckets), duplicates="drop"))
    averages = used_data.groupby(by="x_bin")[[y_var, x_var]].mean()
    counts = used_data.groupby(by="x_bin")[[x_var]].count().rename(columns={x_var: "count"})

    if with_CI:
        # Check first that the dependent variable is 0 and 1 exclusively
        if not all([value in [0,1] for value in list(used_data[y_var].dropna().unique())]):
            raise ValueError('Your y-variable is not exclusively 0 and 1. CI is not implemented for this, use stderr')

        counts_y = used_data.groupby(by="x_bin")[[y_var]].count().rename(columns={y_var: "count_y"})
        averages = pd.concat([averages, counts, counts_y], axis = 1)

        # This assumes that the actual probability is true.
        # Alternative method is to use statsmodels.stats.proportion.proportion_confint
        plot_dataset = pd.concat([
            averages,
            pd.DataFrame({"lower_bound": averages.apply(lambda x: ss.binom.ppf(lower, x["count_y"], x[y_var]) / x["count_y"], axis=1)}),
            pd.DataFrame({"upper_bound": averages.apply(lambda x: ss.binom.ppf(upper, x["count_y"], x[y_var]) / x["count_y"], axis=1)})
        ], axis = 1)

    elif with_stderr:
        stdevs = used_data.groupby(by="x_bin")[[y_var]].std().rename(columns={y_var: "stdev"})
        plot_dataset = pd.concat([averages, stdevs, counts], axis = 1)
        plot_dataset = plot_dataset.assign(stderr=plot_dataset["stdev"] / np.sqrt(plot_dataset["count"]))

        # Calculate the bounds
        plot_dataset = plot_dataset.assign(lower_bound=np.maximum(lower, plot_dataset[y_var] - plot_dataset["stderr"]))
        plot_dataset = plot_dataset.assign(upper_bound=np.minimum(upper, plot_dataset[y_var] + plot_dataset["stderr"]))

    else:
        plot_dataset = pd.concat([averages, counts], axis=1)

    # Sort dataset
    plot_dataset = plot_dataset.sort_values(by=x_var)

    fig, ax1 = plt.subplots()
    ax1.scatter(x=plot_dataset[x_var].values, y=plot_dataset[y_var].values)
    if with_CI or with_stderr:
        ax1.vlines(plot_dataset[x_var].values,
                   plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values,
                   color="blue", linewidths=1)

    if with_count:
        ax2 = ax1.twinx()
        ax2.fill_between(plot_dataset[x_var], plot_dataset["count"], color="gainsboro")
        ax2.tick_params('y', colors='grey')
        ax2.set_ylabel("# Obs")

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

    if header:
        fig.suptitle(header, fontsize=12, fontweight='bold')
    else:
        fig.suptitle('Bivariate Plot', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax1.set_xlabel(x_var)
    ax1.set_ylabel(y_var)

    # Apply trendline
    if trendline and (x_scale != "symlog") and (y_scale != "symlog"):
        z = np.polyfit(_scaling_functions[x_scale](plot_dataset[x_var].values),
                       _scaling_functions[y_scale](plot_dataset[y_var].values), 1)
        p = np.poly1d(z)
        ax1.plot(plot_dataset[x_var].values,
                 _inverse_scaling_functions[y_scale](p(_scaling_functions[x_scale](plot_dataset[x_var].values))), "r--")

    # Apply the right transformations to axes
    ax1.set_yscale(y_scale)
    ax1.set_xscale(x_scale)


    return fig, plot_dataset


def bivariate_categorical(used_data, y_var, x_var, discrete=True, y_scale="linear",
                          with_count=True, with_stderr=False, with_CI = False, lower=-np.inf, upper=np.inf, header=None):

    """
    Generates bivariate plots for categorical or discrete variables, where data is grouped by each distinct value of the x-variable. Provides option to scale the y-axis and count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param y_var: a single string with the name of the y-variable
    :param x_var: a single string with the name of the x-variable
    :param discrete: a single boolean indicating whether the variable should be treated as discrete (or categorical, if set to False). Discrete variables must be of numeric or similar type, and the plot will automatically set the x-axis tickmarks. The tickmarks for categorical variables will occur at every single value.
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted on the secondary y-axis
    :param with_stderr: Boolean indicating whether standard error bars are added to the plot
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    (Note that this is only valid for dependent variables that are 0 or 1, and will override stderr if used)
    :param lower:
        For with_stderr: Double indicating the lowest value for the error bars (e.g. 0 for binary data)
        For with_CI: Lower probability
    :param upper:
        For with_stderr: Double indicating the highest value for the error bars (e.g. 0 for binary data)
        For with_CI: Upper probability

    :return: Nothing is returned
    """
    averages = used_data.groupby(by=x_var)[[y_var]].mean()
    counts = used_data.groupby(by=x_var)[[x_var]].count().rename(columns={x_var: "count"})

    if with_CI:
        # Check first that the dependent variable is 0 and 1 exclusively
        if not all([value in [0,1] for value in list(used_data[y_var].dropna().unique())]):
            raise ValueError('Your y-variable is not exclusively 0 and 1. CI is not implemented for this, use stderr')

        counts_y = used_data.groupby(by=x_var)[[y_var]].count().rename(columns={y_var: "count_y"})
        averages = pd.concat([averages, counts, counts_y], axis = 1)

        # Proper one is statsmodels.stats.proportion.proportion_confint
        plot_dataset = pd.concat([
            averages,
            pd.DataFrame({"lower_bound": averages.apply(lambda x: ss.binom.ppf(lower, x["count_y"], x[y_var]) / x["count_y"], axis=1)}),
            pd.DataFrame({"upper_bound": averages.apply(lambda x: ss.binom.ppf(upper, x["count_y"], x[y_var]) / x["count_y"], axis=1)})
        ], axis = 1)

    elif with_stderr:
        stdevs = used_data.groupby(by=x_var)[[y_var]].std().rename(columns={y_var: "stdev"})
        plot_dataset = pd.concat([averages, stdevs, counts], axis=1)
        plot_dataset = plot_dataset.assign(stderr=plot_dataset["stdev"] / np.sqrt(plot_dataset["count"]))

        # Calculate the bounds
        plot_dataset = plot_dataset.assign(lower_bound=np.maximum(lower, plot_dataset[y_var] - plot_dataset["stderr"]))
        plot_dataset = plot_dataset.assign(upper_bound=np.minimum(upper, plot_dataset[y_var] + plot_dataset["stderr"]))

    else:
        plot_dataset = pd.concat([averages, counts], axis=1)

    # Sort dataset
    plot_dataset = plot_dataset.sort_index()

    # Generate plots
    fig, ax1 = plt.subplots()
    if discrete:
        ax1.scatter(x=plot_dataset.index.values, y=plot_dataset[y_var].values)
        if with_CI or with_stderr:
            ax1.vlines(plot_dataset.index.values, plot_dataset["lower_bound"].values,
                       plot_dataset["upper_bound"].values, color="blue", linewidths=3)
    else:
        numeric_tickmarks = np.arange(0, plot_dataset.shape[0])
        ax1.scatter(x=numeric_tickmarks, y=plot_dataset[y_var].values)
        if with_CI or with_stderr:
            ax1.vlines(numeric_tickmarks, plot_dataset["lower_bound"].values, plot_dataset["upper_bound"].values,
                       color="blue", linewidths=3)
        plt.setp(ax1.get_xticklabels(), visible=True, rotation=90)
        plt.xticks(numeric_tickmarks, plot_dataset.index.values, size='small')

    if with_count:
        ax2 = ax1.twinx()
        if discrete:
            ax2.fill_between(plot_dataset.index.values, plot_dataset["count"], color="lightgrey")
        else:
            ax2.bar(numeric_tickmarks, plot_dataset["count"], color="lightgrey", width=1)
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
    
    ax1.set_yscale(y_scale)    
    ax1.set_xlabel(x_var)
    ax1.set_ylabel(y_var)
    
    if header:
        fig.suptitle(header, fontsize=12, fontweight='bold')
    else:
        fig.suptitle('Bivariate Plot', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, plot_dataset


def scatterplot_batch(frame, y, all_x, log_y=False, log_x=False, sample_size=1000, save_outputs="sample_outputs", **kwargs):
    
    if save_outputs is not None:
        now = datetime.now() # current date and time
        folder_name = "{} Scatter".format(now.strftime("%Y%m%d %H%M%S"))
        if not os.path.exists(save_outputs):
            os.makedirs(save_outputs)
        output_folder = "{}/{}".format(save_outputs,folder_name)
        os.makedirs(output_folder)
    
    if frame.shape[0] > sample_size:
        plot_data = frame.sample(sample_size)
        print("Sampled {} out of {} observations".format(sample_size, frame.shape[0]))
        
        # Generate min and max
        frame[[y] + all_x].apply(lambda x: np.nanmin(x), axis=1)
        
        summary_table = pd.concat([
            frame[[y] + all_x].apply(lambda x: np.nanmin(x), axis=0),
            frame[[y] + all_x].apply(lambda x: np.nanmax(x), axis=0),
            plot_data[[y] + all_x].apply(lambda x: np.nanmin(x), axis=0),
            plot_data[[y] + all_x].apply(lambda x: np.nanmax(x), axis=0),
        ],axis=1).reset_index()
        
        summary_table.columns = ["Variable", "Min", "Max", "Sample Min", "Sample Max"]
        summary_table.to_csv("{}/sample_stat.csv".format(output_folder),index=False)
    else:
        summary_table = None
        plot_data = frame
    
    if log_y:
        y_data = _log1p_abs(plot_data[y])
        y_label = "signed log of {}".format(y)
    else:
        y_data = plot_data[y]
        y_label = y
    
    fig_dict = {}
    for x_variable in all_x:
        
        fig, ax = plt.subplots()
        
        if log_x:
            x_data = _log1p_abs(plot_data[x_variable])
            x_label = "signed log of {}".format(x_variable)
        else:
            x_data = plot_data[x_variable]
            x_label = x_variable
        
        ax.scatter(x=x_data, y=y_data, alpha=0.5, s=2)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        
        filename = "{}/{}.png".format(output_folder, x_variable)
        
        plt.savefig(filename, **kwargs)
        fig_dict[x_variable] = ax
    
    return fig_dict, summary_table
    

def kdeplot_num(frame, y, x, log_y=False, log_x=False, sample_size = 1000):

    if frame[[y, x]].isna().any().any():
        print("Removed {} missing".format(frame[[y, x]].isna().any(1).sum()))
        frame = frame[[y, x]].dropna()

    if frame.shape[0] > sample_size:
        print(
        "There are more than {} observations, so a sample will be used to generate the plot".format(sample_size))
        
        pre_sample = frame[[y,x]]
        frame = frame.sample(sample_size)
        
        print("Sampled {} out of {} observations".format(sample_size, frame.shape[0]))
        print("Target variable {} has a range of [{},{}] prior to sampling and a range of [{},{}] following sampling".format(
            y, np.nanmin(pre_sample[y]), np.nanmax(pre_sample[y]), np.nanmin(frame[y]), np.nanmax(frame[y])))
        print("Target variable {} has a range of [{},{}] prior to sampling and a range of [{},{}] following sampling".format(
            x, np.nanmin(pre_sample[x]), np.nanmax(pre_sample[x]), np.nanmin(frame[x]), np.nanmax(frame[x])))
        
    if log_y:
        use_y = _log1p_abs(frame[y])
        y_label = "Signed log transformed {}".format(y)
    else:
        use_y = frame[y]
        y_label = y

    if log_x:
        use_x = _log1p_abs(frame[x])
        x_label = "Signed log transformed {}".format(x)
    else:
        use_x = frame[x]
        x_label = x

    fig = sns.jointplot(y=use_y, x=use_x, kind="kde", fill=True)
    fig.ax_marg_x.set_title('Joint plot of {} and {}'.format(y, x))
    fig.ax_joint.set_ylabel(y_label)
    fig.ax_joint.set_xlabel(x_label)

    return fig


def pairplot_num(frame, var_names, log = False, sample_size = 1000):

    if frame.shape[0] > sample_size:
        frame = frame[var_names].sample(sample_size)

    def special_log_transform(x):
        return np.sign(x) * np.log10(1 + x)

    if log:
        frame = frame[var_names].applymap(special_log_transform)

    # g = sns.PairGrid(frame[var_names])
    # g.map_diag(sns.kdeplot)
    # g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)

    g = sns.pairplot(frame[var_names], dropna=True, diag_kind="kde",kind='reg',
                     plot_kws={'scatter_kws': {'alpha': 0.1}})

    return g


def pairplot_mix(frame, num_vars, cat_vars, log = False, sample_size = 1000):

    if frame.shape[0] > sample_size:
        frame = frame[num_vars + cat_vars].sample(sample_size)
        
    if log:
        frame = pd.concat([frame[num_vars].applymap(_log1p_abs), frame[cat_vars]], axis = 1)

    g = sns.PairGrid(frame, x_vars = cat_vars, y_vars = num_vars, size=10)
    g.map(sns.violinplot, palette="pastel")
    for i, ax in enumerate(g.fig.axes):  ## getting all axes of the fig object
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return g


def crosstab(frame, cat1, cat2, sample_size = 100000, sum_axis = 1):

    if frame.shape[0] > sample_size:
        frame = frame[[cat1, cat2]].sample(sample_size)
    else:
        frame = frame[[cat1, cat2]]

    return pd.crosstab(frame[cat1], frame[cat2], dropna = False).apply(lambda r: r/r.sum(), axis=sum_axis)


def heatmap_crosstab(frame, cat1, cat2, sample_size = 100000, sum_axis = 1):
    g = sns.heatmap(crosstab(frame, cat1, cat2, sample_size, sum_axis))
    return g


def axis_transform(plot_dataset, y_var, y_scale):
    """
           Utility for plotting of model comparison plots

           Args:
               :param plot_dataset: a single pandas DataFrame containing summarized model fit data
               :param y_var: a single string with the name of the dependent variable
               :param y_scale: Optional string of either 'identity' or 'logit' with which to scale the y-axis

           Returns:
               :return: Enriched plot_dataset with transformed variables
    """
    if y_scale == "identity":
        def transformation_function(x):
            return(x)
    elif y_scale == "logit":
        def transformation_function(x):
            return(np.log(x / (1-x)))
    else:
        raise Exception("Error: the selected y_scale is not supported")

    plot_dataset = plot_dataset.assign(trans_y=transformation_function(plot_dataset[y_var]))

    if "lower_bound" in plot_dataset.columns.values:
        plot_dataset.lower_bound.fillna(plot_dataset.trans_y, inplace=True)
        plot_dataset = plot_dataset.assign(lower=transformation_function(plot_dataset.lower_bound))
        plot_dataset.upper_bound.fillna(plot_dataset.trans_y, inplace=True)
        plot_dataset = plot_dataset.assign(upper=transformation_function(plot_dataset.upper_bound))

    return plot_dataset



def segment_bivariate_continuous(used_data, y_var, x_var, segment_var, num_buckets=20, y_scale="linear", x_scale='linear'):

    # Remove missing values of the segment variable
    remove_index = pd.isnull(used_data[segment_var])
    used_data = used_data.loc[~remove_index,:]

    dict_of_segments = {k: v for k, v in used_data.groupby(segment_var)}

    plot_data = None
    for key in dict_of_segments:
        segment_data = dict_of_segments[key]

        temp_plot, temp_data = bivariate_continuous(segment_data, y_var, x_var, num_buckets=num_buckets, with_count = True)

        plt.close()

        temp_data = temp_data[[x_var, y_var, "count"]].reset_index(drop = True)
        temp_data = temp_data.assign(Segment = key)

        if plot_data is None:
            plot_data = temp_data
        else:
            plot_data = pd.concat([plot_data, temp_data])

    fig, ax = plt.subplots(1,2,figsize=(20, 10))
    for label, df in plot_data.groupby('Segment'):
        df.plot(x = x_var, y = y_var, kind="line", ax=ax[0], label=label, marker='o', linestyle = 'None',linewidth=1)
    ax[0].set_title("Bivariate plot for {} by {}".format(x_var, segment_var))
    ax[0].set_xscale(x_scale)    
    ax[0].set_yscale(y_scale)  
    ax[0].set_xlabel(x_var)
    ax[0].set_ylabel(y_var)

    for label, df in plot_data.groupby('Segment'):
        df.plot(x = x_var, y = "count", kind="line", ax=ax[1], label=label, marker='o', linewidth=1)
    ax[1].set_title("Count for {} by {}".format(x_var, segment_var))
    ax[1].set_ylabel("Count")
    plt.legend()

    return fig, plot_data


def segment_bivariate_categorical(used_data, y_var, x_var, segment_var, discrete = True, y_scale="linear"):

    # Remove missing values of the segment variable
    remove_index = pd.isnull(used_data[segment_var])
    used_data = used_data.loc[~remove_index,:]

    dict_of_segments = {k: v for k, v in used_data.groupby(segment_var)}

    plot_data = None
    for key in dict_of_segments:
        segment_data = dict_of_segments[key]

        temp_plot, temp_data = bivariate_categorical(segment_data, y_var, x_var, discrete=discrete,
                                                     y_scale=y_scale, with_count=True)

        plt.close()

        temp_data = temp_data.reset_index()[[x_var, y_var, "count"]]
        temp_data = temp_data.assign(Segment = key)

        if plot_data is None:
            plot_data = temp_data
        else:
            plot_data = pd.concat([plot_data, temp_data])

    segments = plot_data["Segment"].unique()
    x_vals = plot_data[x_var].unique()

    data_structure = pd.DataFrame(np.transpose([np.tile(segments, len(x_vals)), np.repeat(x_vals, len(segments))]))
    data_structure.columns = ["Segment", x_var]

    plot_data = pd.merge(data_structure, plot_data, how='left', on=["Segment", x_var]).sort_values(["Segment", x_var])

    fig, ax = plt.subplots(1,2,figsize=(20, 10))
    for label, df in plot_data.groupby('Segment'):
        df.plot(x = x_var, y = y_var, kind="line", ax=ax[0], label=label, marker='o', linewidth=1)
    ax[0].set_title("Bivariate plot for {} by {}".format(x_var, segment_var))    
    ax[0].set_yscale(y_scale)  
    ax[0].set_xlabel(x_var)
    ax[0].set_ylabel(y_var)
    plt.xticks(rotation=90)

    for label, df in plot_data.groupby('Segment'):
        df.plot(x = x_var, y = "count", kind="line", ax=ax[1], label=label, marker='o', linewidth=1)
    ax[1].set_title("Count for {} by {}".format(x_var, segment_var))
    ax[1].set_ylabel("Count")
    plt.xticks(rotation=90)
    plt.legend()

    return fig, plot_data

