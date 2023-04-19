import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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


def scatter_plot_comparison(dataset, actual, predicted, N=1000, colormap=None, equalize_axes=True, plotsize = (12,5)):

    plot_data = dataset[[actual] + predicted].dropna()
    if plot_data.shape[0] > N:
        plot_data = plot_data.sample(N)
        print("Sampled {} out of {} observations".format(N, dataset.shape[0]))
        
    fig, ax = plt.subplots(figsize=plotsize) 
    
    # Default colormap
    if colormap is None:
        colormap = plt.cm.get_cmap('viridis')
    
    for idx, pred in enumerate(predicted):
        ax.scatter(x=plot_data[pred], y=plot_data[actual], alpha=0.5, s=5, color = colormap(np.linspace(0, 1, len(predicted))[idx]), label = pred)

    # Format the axis
    min_axis = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_axis = max(ax.get_xlim()[1], ax.get_ylim()[1])
    
    if equalize_axes:
        ax.set_xlim(min_axis, max_axis)
        ax.set_ylim(min_axis, max_axis)

    # Add a 45 degree line
    ax.plot([(min_axis, min_axis),(max_axis, max_axis)],
            [(min_axis, min_axis),(max_axis, max_axis)], zorder=0)

    # Label the axes
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    
    # Set legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    
    return fig, ax


def act_vs_pred_comparison(dataset, actual, predicted, num_buckets=20, colormap=None, equalize_axes=True, plotsize = (12,5), non_nan = True):
    """
    Generates plots of actuals vs. predicted grouped by buckets of prediction percentiles. Provides option to include
        count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction

    :return: fig, axs pair
    """

    if non_nan:
        dataset = dataset.dropna(axis = 0, how = 'any', subset = [actual] + predicted)
    
    fig, ax = plt.subplots(figsize=plotsize)
    
    # Default colormap
    if colormap is None:
        colormap = plt.cm.get_cmap('viridis')
    
    plot_dataset = []
    
    for idx, pred in enumerate(predicted):
        
        quantiled = dataset.assign(x_bin=pd.qcut(dataset[pred], int(num_buckets), duplicates="drop"))
        bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual)
        
        plot_data = bounded_dataset \
                        .merge(quantiled.groupby("x_bin")[pred].agg('mean').reset_index(), on="x_bin", how="left") \
                        .rename(columns={"mean": pred}) \
                        .sort_values(by=pred)
        
        plot_dataset.append(plot_data)
        
        ax.scatter(x=plot_data[pred], y=plot_data[actual], marker = "o", alpha=0.5, s=20, color = colormap(np.linspace(0, 1, len(predicted))[idx]), label = pred)
        
    # Add 45 degree line
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, "g-")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actuals')

    # Put a legend to the right of the current axis
    fig.tight_layout()
    
    # Set legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    
    return fig, ax, plot_dataset


def model_comparison_continuous(used_data, actual_var, pred_var, x_var, num_buckets=20, y_scale="linear", x_scale="linear",
                                with_count=True, with_stdev = False, with_CI = False,
                                lower=-np.inf, upper=np.inf, non_nan = True):
    """
    Generates plots of actuals and predicted grouped by buckets of x-variable percentiles. Provides option to scale
    the y-axis and count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param x_var: a single string with the name of the x-variable
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param x_scale: Optional string of either 'linear' or 'logit' with which to scale the x-axis
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations that should be
        plotted on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    (Note that this is only valid for dependent variables that are 0 or 1, and will override stderr if used)
    :param with_stdev: Optional boolean indicating whether error bars representing the standard error of the
        actual dependent variable is included
    :param lower:
        For with_stderr: Double indicating the lowest value for the error bars (e.g. 0 for binary data)
        For with_CI: Lower probability
    :param upper:
        For with_stderr: Double indicating the highest value for the error bars (e.g. 1 for binary data)
        For with_CI: Upper probability
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted
        has missings. If False, then will count observations based on actuals
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis = 0, how = 'any', subset = [actual_var, pred_var])

    quantiled = used_data.assign(x_bin=pd.qcut(used_data[x_var], int(num_buckets), duplicates="drop"))

    if with_CI:
        if not all([value in [0,1] for value in list(used_data[actual_var].dropna().unique())]):
            raise ValueError('Your y-variable is not exclusively 0 and 1. CI is not implemented for this, use stderr')
        bounded_dataset = CI_dataset(quantiled, 'x_bin', actual_var, lower, upper)

    elif with_stdev:
        bounded_dataset = bound_dataset(quantiled, 'x_bin', actual_var, lower, upper)

    else:
        bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual_var)

    plot_dataset = bounded_dataset\
        .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": pred_var}) \
        .merge(quantiled.groupby("x_bin")[x_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": x_var}) \
        .sort_values(by=x_var)

    # Generate plots
    fig, ax1 = plt.subplots()
    act_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[actual_var].values, "bo")
    pred_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[pred_var].values, "ro-")
    if with_stdev or with_CI:
        ax1.vlines(plot_dataset[x_var].values,
                   plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values,
                   color="blue", linewidths=3, zorder=30)
                   
    # Apply the right transformations to axes
    ax1.set_yscale(y_scale)
    ax1.set_xscale(x_scale)
    ax1.set_ylabel(actual_var)
    ax1.set_xlabel(x_var)
    
    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        ax2.vlines(plot_dataset[x_var], 0, plot_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend([act_line, pred_line], ['Actual', 'Predicted'], loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    plt.show()
    

    return fig, axs, plot_dataset


def model_comparison_categorical(used_data, actual_var, pred_var, x_var, discrete=True, y_scale="linear",
                                 with_count=True, with_stdev = False, with_CI = False,
                                 lower=-np.inf, upper=np.inf, non_nan = True):
    """
    Generates plots of actuals and predicted grouped by categorical or discrete variables, where data is grouped by
        each distinct value of the x-variable. Provides option to scale the y-axis and count of observations in each
        bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param x_var: a single string with the name of the x-variable
    :param discrete: a single boolean indicating whether the variable should be treated as discrete (or categorical,
        if set to False). Discrete variables must be of numeric or similar type, and the plot will automatically set
        the x-axis tickmarks. The tickmarks for categorical variables will occur at every single value.
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted
        on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    (Note that this is only valid for dependent variables that are 0 or 1, and will override stderr if used)
    :param with_stdev: Optional boolean indicating whether error bars representing the standard error of the
        actual dependent variable is included
    :param lower:
        For with_stderr: Double indicating the lowest value for the error bars (e.g. 0 for binary data)
        For with_CI: Lower probability
    :param upper:
        For with_stderr: Double indicating the highest value for the error bars (e.g. 1 for binary data)
        For with_CI: Upper probability
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis = 0, how = 'any', subset = [actual_var, pred_var])

    if with_CI:
        if not all([value in [0,1] for value in list(used_data[actual_var].dropna().unique())]):
            raise ValueError('Your y-variable is not exclusively 0 and 1. CI is not implemented for this, use stderr')
        bounded_dataset = CI_dataset(used_data, x_var, actual_var, lower, upper)

    elif with_stdev:
        bounded_dataset = bound_dataset(used_data, x_var, actual_var, lower, upper)

    else:
        bounded_dataset = average_count_dataset(used_data, x_var, actual_var)

    plot_dataset = bounded_dataset \
        .merge(used_data.groupby(x_var)[pred_var].agg('mean').reset_index(), on=x_var, how="left") \
        .rename(columns={"mean": pred_var}) \
        .sort_values(by=x_var)
    
    # Generate plots
    fig, ax1 = plt.subplots()
    if discrete:
        act_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[actual_var].values, "bo")
        pred_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[pred_var].values, "ro-")

        if with_stdev or with_CI:
            ax1.vlines(plot_dataset[x_var].values, plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values, color="blue", linewidths=3)
    else:
        numeric_tickmarks = np.arange(0, plot_dataset.shape[0])
        act_line, = ax1.plot(numeric_tickmarks, plot_dataset[actual_var].values, "bo")
        pred_line, = ax1.plot(numeric_tickmarks, plot_dataset[pred_var].values, "ro-")

        if with_stdev or with_CI:
            ax1.vlines(numeric_tickmarks, plot_dataset["lower_bound"].values, plot_dataset["upper_bound"].values,
                       color="blue", linewidths=3)
        plt.setp(ax1.get_xticklabels(), visible=True, rotation=90)
        plt.xticks(numeric_tickmarks, plot_dataset[x_var].values, size='small')

    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        if discrete:
            ax2.vlines(plot_dataset[x_var], 0, plot_dataset["count"], colors="lightgrey")
        else:
            ax2.vlines(numeric_tickmarks, 0, plot_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        
        axs.append(ax2)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend([act_line, pred_line], ['Actual', 'Predicted'], loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.show()
    return fig, axs, plot_dataset


def bound_dataset(dataset, key, column_name, lower = -np.inf, upper = np.inf):
    """
    Attaches lower and upper bounds for a given column by key

    :param dataset: a single pandas DataFrame containing the y-variable and the key
    :param key: a string representing the name of the group-by key (e.g. x-variable bins)
    :param column_name: a string representing the variable name that should be summarized
    :param lower: (default: -inf) double indicating the lowest value for the error bars (e.g. 0 for binary data)
    :param upper: (default: inf) double indicating the highest value for the error bars (e.g. 1 for binary data)
    :return: a pandas DataFrame that contains the summarized (mean, standard deviation bands, and counts) of the
        dependent variable grouped by the key-variable
    """

    statistics = dataset.groupby(key)[column_name].agg(['mean', 'std', 'count'])\
        .reset_index()

    plot_dataset = statistics.assign(stderr=lambda f: f['std'] / np.sqrt(f['count']))\
        .assign(
            lower_bound=lambda f: np.maximum(lower, f["mean"] - f['stderr']),
            upper_bound=lambda f: np.minimum(upper, f["mean"] + f['stderr']))\
        .rename(columns={"mean": column_name})
    return plot_dataset


def CI_dataset(dataset, key, column_name, lower = 0.05, upper = 0.95):
    """
    Attaches lower and upper bounds for a given column by key

    :param dataset: a single pandas DataFrame containing the y-variable and the key
    :param key: a string representing the name of the group-by key (e.g. x-variable bins)
    :param column_name: a string representing the variable name that should be summarized
    :param lower: double indicating the upper percentile of the CI
    :param upper: double indicating the lower percentile of the CI
    :return: a pandas DataFrame that contains the summarized (mean, standard deviation bands, and counts) of the
        dependent variable grouped by the key-variable
    """

    averages = dataset.groupby(key)[column_name].agg(['mean', 'count'])\
        .rename(columns={'mean': column_name}).reset_index()

    # This assumes that the actual probability is true.
    # Alternative method is to use statsmodels.stats.proportion.proportion_confint
    plot_dataset = pd.concat([
        averages,
        pd.DataFrame({"lower_bound": averages.apply(
            lambda x: ss.binom.ppf(lower, x["count"], x[column_name]) / x["count"], axis=1)}),
        pd.DataFrame({"upper_bound": averages.apply(
            lambda x: ss.binom.ppf(upper, x["count"], x[column_name]) / x["count"], axis=1)})
    ], axis=1)

    return plot_dataset


def average_count_dataset(dataset, key, column_name):
    """
    Calculates averages and counts for a given column by key

    :param dataset: a single pandas DataFrame containing the y-variable and the key
    :param key: a string representing the name of the group-by key (e.g. x-variable bins)
    :param column_name: a list of variable names that should be summarized
    :return: a pandas DataFrame that contains the summarized (mean, standard deviation bands, and counts) of the
        dependent variable grouped by the key-variable
    """

    plot_dataset = dataset.groupby(key)[column_name].agg(['mean', 'count'])\
        .reset_index() \
        .rename(columns={"mean": column_name})

    return plot_dataset