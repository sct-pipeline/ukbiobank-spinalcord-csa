#!/usr/bin/env python
# -*- coding: utf-8
# Computes statistical analysis for ukbiobank-spinalcord-csa
#
# For usage, type: uk_compute_stats -h

# Authors: Sandrine Bédard

import os
import argparse
import pandas as pd
import numpy as np
import logging
import sys
import yaml
import scipy
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from textwrap import dedent
from sklearn.preprocessing import PolynomialFeatures

FNAME_LOG = 'log_stats.txt'

plt.style.use('seaborn')

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

# Dictionnary of the predictors with units
PREDICTORS = {
    'sex': '',
    'height': 'cm',
    'weight': 'kg',
    'age': 'year',
    'Vscale': '',
    'ventricular CSF volume': 'mm^2',
    'brain GM volume': 'mm^2',
    'brain WM volume': 'mm^2',
    'brain volume norm': '',
    'brain volume': 'mm^2',
    'thalamus volume': 'mm^2'
    }

MODELS = {
    'model_1': ['thalamus volume', 'brain volume', 'height', 'sex', 'age', 'weight', 'ventricular CSF volume'],
    'model_BV_sex': ['brain volume', 'sex', 'inter BV_sex'],
    'model_BV_TV_sex': ['sex', 'brain volume', 'thalamus volume', 'inter BV_sex'],
}


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes the statistical analysis with the results of the ukbiobank project",
        prog=os.path.basename(__file__).strip('.py'),
        formatter_class=SmartFormatter
        )
    parser.add_argument('-path-output',
                        required=False,
                        type=str,
                        metavar='<dir_path>',
                        help="Path to folder that contains output files (processed_data, log, qc, results)")
    parser.add_argument('-dataFile',
                        required=False,
                        default='data_ukbiobank.csv',
                        metavar='<file>',
                        help="Filename of the output file of uk_get_subject_info Default: data_ukbiobank.csv  ")
    parser.add_argument('-output-name',
                        required=False,
                        default='stats_results',
                        metavar='<folder>',
                        help="Name of output folder with results of statistics.")
    parser.add_argument('-exclude',
                        metavar='<file>',
                        required=False,
                        help=
                        "R|Config yaml file listing subjects to exclude from statistical analysis.\n"
                        "Yaml file can be validated at this website: http://www.yamllint.com/.\n"
                        "Below is an example yaml file:\n"
                        + dedent(
                                 """
                                 - sub-1000032_T1w.nii.gz
                                 - sub-1000498_T1w.nii.gz
                                 """)
                        )
    return parser


def compute_statistics(df):
    """
    Compute statistics such as mean, std, COV, etc. of CSA values.
    Args:
        df (panda.DataFrame): dataframe of all parameters and CSA values
    Returns:
        stats_df (panda.DataFrame): statistics of CSA per contrast type
    """
    contrasts = ['CSA_c2c3', 'CSA_pmj']
    metrics = ['n', 'mean', 'std', 'med', '95ci', 'COV', 'max', 'min', 'normality_test_p']
    stats = {}
    for contrast in contrasts:
        stats[contrast] = {}
        for metric in metrics:
            stats[contrast][metric] = {}
    # Computes the metrics
    for contrast in contrasts:
        stats[contrast]['n'] = len(df[contrast])
        stats[contrast]['mean'] = np.mean(df[contrast])
        stats[contrast]['std'] = np.std(df[contrast])
        stats[contrast]['med'] = np.median(df[contrast])
        stats[contrast]['95ci'] = 1.96*np.std(df[contrast])/np.sqrt(len(df[contrast]))
        stats[contrast]['COV'] = np.std(df[contrast]) / np.mean(df[contrast])
        stats[contrast]['max'] = np.max(df[contrast])
        stats[contrast]['min'] = np.min(df[contrast])
        # Validate normality of CSA with Shapiro-wilik test
        stats[contrast]['normality_test_p'] = scipy.stats.shapiro(df[contrast])[1]
        # Writes a text with CSA stats
        output_text_CSA_stats(stats, contrast)

    # Convert dict to DataFrame
    stats_df = pd.DataFrame.from_dict(stats)
    return stats_df


def output_text_CSA_stats(stats, contrast):
    """
    Embed statistical results of CSA into sentences so they can easily be copy/pasted into a manuscript.
    Args:
        stats (dict): dictionnary with stats of the predictors for one contrast
    """
    logger.info('\nStatistics of {}:'.format(contrast))
    logger.info('   There are {} subjects included in the analysis.'.format(stats[contrast]['n']))
    logger.info('   CSA values are between {:.6} and {:.6} mm^2'.format(stats[contrast]['min'], stats[contrast]['max']))
    logger.info('   Mean CSA is {:.6} mm^2, standard deviation CSA is of {:.6}, median value is of {:.6} mm^2.'.format(stats[contrast]['mean'], stats[contrast]['std'], stats[contrast]['med']))
    logger.info('   The COV is of {:.6} and 95 confidence interval is {:.6}.'.format(stats[contrast]['COV'], stats[contrast]['95ci']))
    logger.info('   The results of Shapiro-wilik test has a p-value of {:.6}.'.format(stats[contrast]['normality_test_p']))


def compute_predictors_statistic(df):
    """
    Compute statistics such as mean, min, max for each predictor and writes it in the log file.
    Args:
        df (panda.DataFrame): Dataframe of all predictor and CSA values
    Returns:
        stats_df (panda.DataFrame): Statistics of each predictors
    """
    stats = {}
    metrics = ['min', 'max', 'med', 'mean']
    for predictor in PREDICTORS.keys():
        stats[predictor] = {}
        for metric in metrics:
            stats[predictor][metric] = {}
    for predictor in PREDICTORS.keys():
        stats[predictor]['min'] = np.min(df[predictor])
        stats[predictor]['max'] = np.max(df[predictor])
        stats[predictor]['med'] = np.median(df[predictor])
        stats[predictor]['mean'] = np.mean(df[predictor])
    # Computes male vs female ratio
    stats['sex'] = {}
    stats['sex']['%_M'] = 100*(np.count_nonzero(df['sex']) / len(df['sex']))
    stats['sex']['%_F'] = 100 - stats['sex']['%_M']
    # Writes statistics of predictor into a text
    output_text_stats(stats)
    stats_df = pd.DataFrame.from_dict(stats)  # Converts dict to dataFrame

    return stats_df


def output_text_stats(stats):
    """
    Embed statistical results of predictors into sentences so they can easily be copy/pasted into a manuscript.
    Args:
        stats (dict): dictionnary with stats of the predictors
    """
    logger.info('sex statistic:')
    logger.info('   {:.3} % of male  and {:.3} % of female.'.format(stats['sex']['%_M'],
                                                                    stats['sex']['%_F']))
    for predictor in [*PREDICTORS][1:]:
        logger.info('{} statistic:'.format(predictor))
        logger.info('   {} between {} and {} {}, median {} {} {}, mean {} {:.6} {}.'.format(predictor,
                                                                                            stats[predictor]['min'],
                                                                                            stats[predictor]['max'],
                                                                                            PREDICTORS[predictor],
                                                                                            predictor,
                                                                                            stats[predictor]['med'],
                                                                                            PREDICTORS[predictor],
                                                                                            predictor,
                                                                                            stats[predictor]['mean'],
                                                                                            PREDICTORS[predictor]))


def format_number(number):
    """
    Round number to two decimals
    :param number: input number
    :return: number rounded to two decimals
    """
    return format(float(number), '.2f')


def scatter_plot(x, y, filename, path, xlabel, ylabel='CSA (mm$^2$)', title=None):
    """
    Generate and save a scatter plot of y and x.
    Args:
        x (panda.DataFrame):
        y (panda.DataFrame):
    """
    plt.figure()
    sns.regplot(x=x, y=y, line_kws={"color": "crimson"})
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    else:
        plt.title('CSA as a function of ' + filename)
    plt.savefig(os.path.join(path, filename + '.png'))
    plt.close()


def scatter_plot_pmj_c2c3(x, y, distance, path):
    """
    Generate and save a scatter plots of y and x, for CSA_pmj and CSA_c2c3 and for distance between C2-C3 and PMJ
        x (panda.DataFrame): C2-C3 CSA
        y (panda.DataFrame): PMJ CSA
        distance (panda.DataFrame): Distance bewteen C2-C3 disc and PMJ
    """
    plt.figure()
    fig, ax = plt.subplots(1, 2)
    sns.scatterplot(ax=ax[0], x=x, y=y, alpha=0.7, edgecolors=None, linewidth=0)
    ax[0].set_xlim(45, 100)
    ax[0].set_ylim(45, 100)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_ylabel('PMJ CSA (mm$^2$)')
    ax[0].set_xlabel('C2-C3 CSA (mm$^2$)')
    ax[0].set_title('a) CSA agreement between PMJ and C2-C3', pad=15)
    # Compute linear fit
    model = generate_linear_model(x, y)
    # Place regression equation to upper-left corner
    ax[0].text(0.1, 0.9, 'y = {}x + {}\nR\u00b2 = {}'.format(format_number(model.params[1]),
                                                             format_number(model.params[0]),
                                                             format_number(model.rsquared)),
               ha='left', va='center', color='crimson', transform=ax[0].transAxes,
               fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=1))  # box around equation
    # Plot linear regression
    x_vals = np.array(ax[0].get_xlim())
    y_vals = model.params[0] + model.params[1] * x_vals
    y_vals = np.squeeze(y_vals)  # change shape from (1,N) to (N,)
    ax[0].plot(x_vals, y_vals, color='crimson', alpha=0.9)
    ax[0].plot([45, 100], [45, 100], ls="--", c="k")  # add diagonal line

    # Scatterplot of distance between PMJ and C2-C3 disc
    mean = np.mean(distance)
    std = np.std(distance)
    sns.scatterplot(ax=ax[1], data=distance, alpha=0.7, edgecolors=None, linewidth=0)
    ax[1].axhline(y=mean, linewidth=2, color='k', ls="--")
    ax[1].set_ylabel('Distance (mm)')
    ax[1].set_xlabel('Participants')
    ax[1].set_title('b) Distance from PMJ and C2-C3', pad=15)
    ax[1].set_box_aspect(1)
    plt.tight_layout()
    logger.info('Mean distance from PMJ to C2-C3 disc is {} mm and standard deviation is {} mm'.format(format_number(mean), format_number(std)))
    filename = 'scatterplots_c2c3_pmj_csa.png'
    plt.savefig(os.path.join(path, filename))
    logger.info('Created: ' + filename)
    plt.close()


def df_to_csv(df, filename):
    """
    Save a Dataframe as a .csv file.
    Args:
        df (panda.DataFrame)
        filename (str): Name of the output .csv file.
    """
    df.to_csv(filename)
    logger.info('Created: ' + filename)


def get_correlation_table(df):
    """
    Return correlation matrix of a DataFrame using Pearson's correlation coefficient, p-values of correlation coefficients and correlation matrix with level of significance *.
    Args:
        df (panda.DataFrame)
    Returns:
        corr_table (panda.DataFrame): correlation matrix of df
        corr_table_pvalue (panda.DataFrame): p-values of correlation matrix of df
        corr_table_and_p_value (panda.DataFrame): correlation matrix of df with level of significance labeled as *, ** or ***
    """
    # TODO: remove half
    corr_table = df.corr(method='pearson')
    corr_table_pvalue = df.corr(method=lambda x, y: scipy.stats.pearsonr(x, y)[1]) - np.eye(len(df.columns))
    # Overcome smallest possible 64bit floating point
    for column in corr_table_pvalue.columns:
        for index in corr_table_pvalue.index:
            if column != index and corr_table_pvalue.loc[index, column] == 0:
                corr_table_pvalue.loc[index, column] = 1e-30
    p = corr_table_pvalue.applymap(lambda x: ''.join(['*' for t in [0.001, 0.05, 0.01] if x <= t and x > 0]))
    corr_table_and_p_value = corr_table.round(2).astype(str) + p
    return corr_table, corr_table_pvalue, corr_table_and_p_value


def compare_sex(df, path):
    """
    Compute mean CSA and std value for each sex, T-test for the means of two independent samples of scores and generates violin plots of CSA for sex.
    Args:
        df (panda.DataFrame)
    """
    # Compute mean CSA value
    mean_csa_F = np.mean(df[df['sex'] == 0]['CSA_pmj'])
    std_csa_F = np.std(df[df['sex'] == 0]['CSA_pmj'])
    mean_csa_M = np.mean(df[df['sex'] == 1]['CSA_pmj'])
    std_csa_M = np.std(df[df['sex'] == 1]['CSA_pmj'])

    # Compute T-test
    results = scipy.stats.ttest_ind(df[df['sex'] == 0]['CSA_pmj'], df[df['sex'] == 1]['CSA_pmj'])
    df_copy = df.copy()
    df_copy.loc[df_copy['sex'] == 0, 'sex'] = 'F'
    df_copy.loc[df_copy['sex'] == 1, 'sex'] = 'M'
    # Violin plot
    plt.figure()
    fig, ax = plt.subplots()
    plt.title("Violin plot of CSA and sex")
    sns.violinplot(y='CSA_pmj', x='sex', data=df_copy, palette='flare')
    # Add mean CSA and std for female
    textstr_F = '\n'.join((
                           r'$\mu=%.2f$' % (mean_csa_F, ),
                           r'$\sigma=%.2f$' % (std_csa_F, )))
    ax.text(0.05, 0.92, textstr_F, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left')
    # Add mean CSA and std for female
    textstr_M = '\n'.join((r'$\mu=%.2f$' % (mean_csa_M, ),
                           r'$\sigma=%.2f$' % (std_csa_M, )))
    ax.text(0.82, 0.92, textstr_M, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left')
    plt.ylabel('CSA (mm$^2$)')
    fname_fig = os.path.join(path, 'violin_plot.png')
    plt.savefig(fname_fig)

    # Write results
    logger.info('Created: ' + fname_fig)
    logger.info('Mean CSA value for female : {:.4} mm^2, std is {:.4}'.format(mean_csa_F, std_csa_F))
    logger.info('Mean CSA value for male : {:.4} mm^2, std is {:.4}'.format(mean_csa_M, std_csa_M))
    logger.info("T test p_value : {} , t value {}".format(results[1], results[0]))


def generate_linear_model(x, y, selected_predictors=None):
    """
    Compute linear regresion with the selected predictors.
    Args:
        x (panda.DataFrame): Data of the predictors
        y (panda.DataFrame): Data of CSA
        selected_predictors (list): List of predictors to inlcude in the linear regression model.
    Returns:
        results (statsmodels.regression.linear_model.RegressionResults object): fitted linear model.
    """
    # Updates x to only have the data of the selected predictors.
    if selected_predictors:
        x = x[selected_predictors]
    # Adds a columns of ones to the original DataFrame.
    x = sm.add_constant(x)

    model = sm.OLS(y, x)  # Computes linear regression
    results = model.fit()  # Fits model
    return results


def analyse_age(x, y, path, lin_model, degree=2):
    """
    Computes quadratic fit, saves the model and plot.
    Args:
        x (panda.DataFrame): Data of the predictors
        y (panda.DataFrame): Data of CSA
    """
    x = np.array(x)
    y = np.array(y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    inds = x.ravel().argsort()
    x = x.ravel()[inds].reshape(-1, 1)
    y = y[inds]
    polynomial_features = PolynomialFeatures(degree)
    xp = polynomial_features.fit_transform(x)
    model = sm.OLS(y, xp).fit()
    ypred = model.predict(xp)
    save_model(model, 'quadratic_fit', path)
    # Scatter plot and quadratic fit
    plt.figure()
    fig, ax = plt.subplots()
    plt.title("CSA as function of age and quadratic fit")
    plt.xlabel('age')
    plt.ylabel('CSA (mm$^2$)')
    plt.scatter(x, y)
    plt.plot(x, ypred, color='crimson')
    plt.text(0.95, 0.9, 'y = {} - {:.2e}x - {:.2e}x$^2$\nR\u00b2 = {:.2}'.format(format_number(model.params[0]),
                                                                     model.params[1]*-1,
                                                                     model.params[2]*-1,
                                                                     model.rsquared),
             ha='right', va='center', color='crimson', transform=ax.transAxes,
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=1))  # box around equation
    fname_fig = os.path.join(path, 'quadratic_fit.png')
    plt.savefig(fname_fig)
    plt.close()
    logger.info('Created: ' + fname_fig)

    # Scatterplot and linear model
    plt.figure()
    fig, ax = plt.subplots()
    plt.title("CSA as function of age and linear fit")
    plt.xlabel('age')
    plt.ylabel('CSA (mm$^2$)')
    plt.scatter(x, y)
    y_vals = lin_model.params[0] + lin_model.params[1] * x
    y_vals = np.squeeze(y_vals)  # change shape from (1,N) to (N,)
    plt.plot(x, y_vals, color='crimson', alpha=0.9)
    plt.text(0.95, 0.9, 'y = {} - {:.3}x\nR\u00b2 = {:.2}'.format(format_number(model.params[0]),
                                                                   model.params[1]*-1,
                                                                   model.rsquared),
             ha='right', va='center', color='crimson', transform=ax.transAxes,
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=1))  # box around equation
    fname_fig = os.path.join(path, 'linear_fit.png')
    plt.savefig(fname_fig)
    plt.close()
    logger.info('Created: ' + fname_fig)


def compute_stepwise(x, y, threshold_in, threshold_out):
    """
    Perform backward and forward predictor selection based on p-values.

    Args:
        x (panda.DataFrame): Candidate predictors
        y (panda.DataFrame): Candidate predictors with target
        threshold_in: include a predictor if its p-value < threshold_in
        threshold_out: exclude a predictor if its p-value > threshold_out
        ** threshold_in <= threshold_out
    Returns:
        included: list of selected predictor

    """
    included = []  # Initialize a list for inlcuded predictors in the model
    while True:
        changed = False
        # Forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_predictor = excluded[new_pval.argmin()]  # Gets the predictor with the lowest p_value
            included.append(best_predictor)  # Adds best predictor to included predictor list
            changed = True
            logger.info('Add  {:30} with p-value {:.6}'.format(best_predictor, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()  # Computes linear regression with included predictor
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # Gets the worst p-value of the model
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_predictor = included[pvalues.argmax()]  # gets the predictor with worst p-value
            included.remove(worst_predictor)  # Removes the worst predictor of included predictor list
            logger.info('Drop {:30} with p-value {:.6}'.format(worst_predictor, worst_pval))
            if worst_pval == best_pval and worst_predictor == best_predictor:  # If inclusion of a paremeter doesn't change p_value, end stepwise to avoid infinite loop
                break
        if not changed:
            break

    return included


def save_model(model, model_name, path_model_contrast, x=None):
    """
    Save summary in .txt file and coeff in a .csv file.

    Args:
        model (statsmodels.regression.linear_model.RegressionResults object):  fited linear model.
        model_name (str): Name of the model
        path_model_contrast (str): Path of the result folder for this model and contrast
    """
    logger.info('Saving {} ...'.format(model_name))

    def save_summary():
        summary_path = os.path.join(path_model_contrast, 'summary')
        # Creates if doesn't exist a folder for the model summary
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_filename = os.path.join(summary_path, 'summary_' + model_name + '.txt')
        # Saves the summary of the model in a .txt file
        with open(summary_filename, 'w') as fh:
            fh.write(model.summary(title=model_name).as_text())
        logger.info('Created: ' + summary_filename)

    def save_coeff():
        coeff_path = os.path.join(path_model_contrast, 'coeff')
        # Creates a folder for results coeff of the model if doesn't exists
        if not os.path.exists(coeff_path):
            os.makedirs(coeff_path)
        coeff_filename = os.path.join(coeff_path, 'coeff_' + model_name + '.csv')
        # Saves the coefficients of the model in .csv file
        logger.info('P_values are : {}'.format(model.pvalues))
        df = pd.DataFrame(model.params, columns=['coeff'])
        if x is not None:
            df.loc[x.columns, 'mean'] = np.mean(x, axis=0)
        df.to_csv(coeff_filename)
        logger.info('Created: ' + coeff_filename)

    save_coeff()
    save_summary()


def compute_regression_csa(x, y, p_in, p_out, contrast, path_model):
    """
    Compute stepwise model and complete linear model of CSA. Save both models, compare and analyse residuals. Apply normalization method from model and compute COV.
    Args:
        x (panda.DataFrame): Data of predictors
        y (panda.DataFrame): Data of CSA
        p_in (float): include a predictor if its p-value < p_in for stepwise ** p_in <= p_out
        p_out (float): exclude a predictor if its p-value > p_out for stepwise
        contrast (str): Contrast of the image that CSA value was computed from
        path_model (str): Path of the result folder of the models
    Return:
        COV_step, COV_full
    """
    # Creates directory for results of CSA model for this contrast if doesn't exists
    path_model_contrast = os.path.join(path_model, contrast)
    if not os.path.exists(path_model_contrast):
        os.mkdir(path_model_contrast)

    # Computes stepwise linear regression with p_value
    logger.info("Stepwise linear regression {}:".format(contrast))
    selected_predictors = compute_stepwise(x, y, p_in, p_out)
    logger.info('For ' + contrast + ' selected predictors are : {}'.format(selected_predictors))

    # Generates model with selected predictors from stepwise
    model = generate_linear_model(x, y, selected_predictors)

    # Compute VIF
    X = sm.add_constant(x[selected_predictors])
    vif_data = pd.Series([variance_inflation_factor(X.values, i)
                         for i in range(X.shape[1])],
                         index=X.columns)
    logger.info('VIF of predictors: \n{}'.format(vif_data))

    # Apply normalization method
    COV_step = apply_normalization(y, x, model.params)
    m1_name = 'stepwise_' + contrast
    # Saves summary of the model and the coefficients of the regression
    save_model(model, m1_name, path_model_contrast, x=x[selected_predictors])

    # Generates linear regression with all predictors
    model_full = generate_linear_model(x, y)
    m2_name = 'fullLin_' + contrast

    # Compute VIF
    X = sm.add_constant(x)
    vif_data_full = pd.Series([variance_inflation_factor(X.values, i)
                               for i in range(X.shape[1])],
                              index=X.columns)
    logger.info('VIF of predictors: \n{}'.format(vif_data_full))

    # Apply normalization method
    COV_full = apply_normalization(y, x, model_full.params)

    # Saves summary of the model and the coefficients of the regression
    save_model(model_full, m2_name, path_model_contrast, x=x)

    # Compares full and reduced models with F_value, R^2,...
    compared_models = compare_models(model, model_full, m1_name, m2_name)
    logger.info('Comparing models: {}'.format(compared_models))
    compared_models_filename = os.path.join(path_model_contrast, 'compared_models') + '.csv'
    df_to_csv(compared_models, compared_models_filename)  # Saves to .csv

    # Residual analysis
    logger.info('\nAnalysing residuals...')
    analyse_residuals(model, m1_name, data=pd.concat([x, y], axis=1), path=os.path.join(path_model_contrast, 'residuals'))
    analyse_residuals(model_full, m2_name, data=pd.concat([x, y], axis=1), path=os.path.join(path_model_contrast, 'residuals'))

    return COV_step, COV_full


def compare_models(model_1, model_2, model_1_name, model_2_name):
    """
    Create a dataframe with R^2, R^2 adjusted, F p_value, F_value, AIC and df of residuals for both models.
    Args:
        model_1 (statsmodels.regression.linear_model.RegressionResults object): First fitted model to compare
        model_2 (statsmodels.regression.linear_model.RegressionResults object): Second fitted model to compare
        model_1_name (str): Name of the first model
        model_2_name (str): Name of the second model
    Returns:
        table (panda.DataFrame) : Dataframe with results of both models
    """
    columns = ['Model', 'R^2', 'R^2_adj', 'F_p-value', 'F_value', 'AIC', 'df_res']
    # Initialize a dataframe
    table = pd.DataFrame(columns=columns)
    table['Model'] = [model_1_name, model_2_name]
    table['R^2'] = [model_1.rsquared, model_2.rsquared]
    table['R^2_adj'] = [model_1.rsquared_adj, model_2.rsquared_adj]
    table['F_p-value'] = [model_1.f_pvalue, model_2.f_pvalue]
    table['F_value'] = [model_1.fvalue, model_2.fvalue]
    table['AIC'] = [model_1.aic, model_2.aic]
    table['df_res'] = [model_1.df_resid, model_2.df_resid]
    table = table.set_index('Model')  # Set index to model names
    return table


def analyse_residuals(model, model_name, data, path):  # TODO: Add residuals as function of each parameter
    """
    Generate and save Residuals vs Fitted values plot and QQ plot.
    Args:
        model (statsmodels.regression.linear_model.RegressionResults object): fitted model to analyse residuals
        model_name (str): Name of the model
        data (panda.DataFrame): Data of all predictors and CSA values.
        path (str): Path to folder of the results of the residuals analysis.
    """
    # Get the residuals from the fitted model
    residual = model.resid
    # Initialize a plot
    plt.figure()
    fig, axis = plt.subplots(1, 2, figsize=(12, 4))
    plt.autoscale(1)
    axis[0].title.set_text('Quantile-quantile plot of residuals')
    axis[1].title.set_text('Residuals vs Fitted')

    # Generate graph of QQ plot | Validate normality hypothesis of residuals
    axis[0] = sm.qqplot(residual, line='s', ax=axis[0])
    # Residual vs fitted values plot
    model_fitted_y = model.fittedvalues
    # Compute plot
    axis[1] = sns.residplot(x=model_fitted_y, y=data.columns[-1], data=data,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # Set axis of Residual vs Fitted values plot
    axis[1].set_xlabel('Fitted values')
    axis[1].set_ylabel('Residuals')

    # Add annotations for the residual far from center
    model_abs_resid = np.abs(residual)
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        axis[1].annotate(i, xy=(model_fitted_y[i], residual[i]))

    # Title of the plot
    fig.suptitle(' Residual analysis of ' + model_name, fontsize=16)
    plt.tight_layout()
    # Create path to folder of the results of the residuals analysis if doesn't exists.
    if not os.path.exists(path):
        os.mkdir(path)
    fname_fig = os.path.join(path + '/res_plots_' + model_name + '.png')
    plt.savefig(fname_fig)  # save plot
    plt.close()
    logger.info('Created: ' + fname_fig)


def apply_normalization(csa, data_predictor, coeff):
    """
    Normalize CSA values with coeff from multivariate model and computes COV.

    Args:
        csa (panda.DataFrame): csa values
        data_predictor (panda.DataFrame): values for all subjects
        coeff (panda.DataFrame): coefficients from multivariate model
    Retruns:
        pred_csa (np.ndarray): predictied CSA values

    """
    coeff.drop('const', inplace=True)
    predictors = list(coeff.index)
    pred_csa = csa.to_numpy()
    for predictor in predictors:
        pred_csa = pred_csa + coeff[predictor]*(data_predictor[predictor].mean() - data_predictor[predictor])
    COV_pred = np.std(pred_csa) / np.mean(pred_csa)
    logger.info('\n COV of normalized CSA: {}'.format(COV_pred))

    return COV_pred


def remove_subjects(df, dict_exclude_subj):
    """
    Remove subjects from exclude list if given and all subjects that are missing a parameter.
    Writes in log the list of removed subjects.
    Args:
        df (panda.DataFrame): Dataframe with all subjects parameters and CSA values.
    Returns
        df_updated (panda.DataFrame): Dataframe without exluded subjects.
    """
    # Initalize list of subjects to exclude with all subjects missing a value
    subjects_removed = df.loc[pd.isnull(df).any(1), :].index.values
    # Remove all subjects passed from the exclude list
    for sub in dict_exclude_subj:
        # Check if contrast is T1w
        if sub[-10:-7] == 'T1w':
            sub_id = (sub[:-11])
            # Check if the subjects is in the dataframe
            if sub_id in df.index:
                df = df.drop(index=sub_id)
                subjects_removed = np.append(subjects_removed, sub_id)  # add subject to excluded list
    # Add subjects with nervous system disorders to list subjects_removed
    subjects_removed = np.append(subjects_removed, df[df['neuro_disease'] == 1].index.values)
    df_updated = df.dropna(0, how='any').reset_index(drop=True)  # Drops all subjects missing a parameter
    df_updated = df_updated[df_updated['neuro_disease'] == 0]  # Keep subject without nervous system disorders
    df_updated.drop('neuro_disease', axis='columns', inplace=True)
    logger.info("{} Subjects removed : {}".format(len(subjects_removed), subjects_removed))
    return df_updated


def init_path_results(path_statistics):
    """
    Create folders for stats results if does not exists.
    Returns:
        path_metric: path to folder of metrics
        path_model: path to folder of stats models
    """   
    if not os.path.exists(path_statistics):
        os.mkdir(path_statistics)
    path_metrics = os.path.join(path_statistics, 'metrics')
    if not os.path.exists(path_metrics):
        os.mkdir(path_metrics)
    path_model = os.path.join(path_statistics, 'models')
    if not os.path.exists(path_model):
        os.mkdir(path_model)
    return (path_metrics, path_model)


def main():

    parser = get_parser()
    args = parser.parse_args()

    # If argument path-ouput included, go to the results folder
    if args.path_output is not None:
        path_results = os.path.join(args.path_output, 'results')
        os.chdir(path_results)

    # Create a panda dataFrame from datafile input arg .csv
    df = (pd.read_csv(args.dataFile)).set_index('Subject')
    # Create a dict with subjects to exclude if input .yml config file is passed
    if args.exclude is not None:
        # Check if input yml file exists
        if os.path.isfile(args.exclude):
            fname_yml = args.exclude
        else:
            sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(args.exclude))
        with open(fname_yml, 'r') as stream:
            try:
                dict_exclude_subj = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
    else:
        # Initialize empty dict if no config yml file is passed
        dict_exclude_subj = dict()

    # Initialize path of statistical results
    path_metrics, path_model = init_path_results(args.output_name)

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), args.output_name, FNAME_LOG))
    logging.root.addHandler(fh)

    # Remove all subjects that are missing a parameter or CSA value, subjects from exclude list and subjects with nervous system disorders.
    df = remove_subjects(df, dict_exclude_subj)

    # Compute stats for CSA_c2c3 and CSA_pmj
    stats_csa = compute_statistics(df)
    # Format and save CSA stats as a.csv file
    metric_csa_filename = os.path.join(path_metrics, 'stats_csa')
    df_to_csv(stats_csa, metric_csa_filename + '.csv')

    # Compute stats of the predictors
    stats_predictors = compute_predictors_statistic(df)
    # Format and save stats of csa as a .csv
    stats_predictors_filename = os.path.join(path_metrics, 'stats_param')
    df_to_csv(stats_predictors, stats_predictors_filename + '.csv')

    # Correlation matrix (Pearson's correlation coefficients)
    corr_table, corr_pvalue, corr_and_pvalue = get_correlation_table(df)
    logger.info("Correlation matrix: {}".format(corr_table))
    corr_filename = os.path.join(path_metrics, 'corr_table')
    # Save a.csv file of the correlation matrix in the results folder
    df_to_csv(corr_table, corr_filename + '.csv')
    df_to_csv(corr_pvalue, corr_filename + '_pvalue.csv')
    df_to_csv(corr_and_pvalue, corr_filename + '_and_pvalue.csv')

    # Generate scatter plot of CSA_pmj and CSA_c2c3 and of distance between C2-C3 disc and PMJ
    path_scatter_plot_c2c3_pmj = os.path.join(path_metrics, 'comparison_c2c3_pmj')
    if not os.path.exists(path_scatter_plot_c2c3_pmj):
        os.mkdir(path_scatter_plot_c2c3_pmj)
    scatter_plot_pmj_c2c3(df['CSA_pmj'], df['CSA_c2c3'], df['distance_c2c3_pmj'], path_scatter_plot_c2c3_pmj)

    # Stepwise linear regression and complete linear regression for PMJ-based CSA
    x = df.drop(columns=['CSA_c2c3', 'CSA_pmj'])  # Initialize x to data of predictors
    y = df['CSA_pmj']

    # Generate scatter plots for all predictors and CSA
    path_scatter_plots = os.path.join(path_metrics, 'scatter_plots')
    if not os.path.exists(path_scatter_plots):
        os.mkdir(path_scatter_plots)
    for column, data in x.iteritems():
        scatter_plot(data, y, column, path_scatter_plots, xlabel=column)

    # Generate scatterplot total brain volume vs age
    scatter_plot(x['age'], x['brain volume'], 'scatter_plot_age_brain_vol', path_metrics, xlabel='age', ylabel='Total Brain Volume (mm$^3$)', title='Total Brain Volume as a function of age')
    scatter_plot(x['age'], x['thalamus volume'], 'scatter_plot_age_thalamus_vol', path_metrics, xlabel='age', ylabel='thalamus volume (mm$^3$)', title='thalamus volume as a function of age')

    # Create pairwise plot between CSA and preditors seprated for sex | Maybe not useful, but nice to see
    plt.figure()
    g = sns.pairplot(df, x_vars=['brain volume','thalamus volume'], y_vars='CSA_pmj', kind='reg', hue='sex', 
                 palette="Set1",  height = 4, plot_kws={'scatter_kws': {'alpha': 0.6}, 'line_kws':{'lw':4}})
    g.axes[0,0].yaxis.set_label_text('CSA (mm$^2$)')
    g.fig.suptitle('Scatterplots of CSA as a function of brain volume and thalamus volume')
    g._legend.remove()
    plt.legend(title='sex', loc='lower right', labels=['F', 'M'])
    plt.tight_layout()
    plt.savefig(os.path.join(path_scatter_plots, 'pairwise_plot' + '.png'))
    plt.close()
    logger.info("Created Scatter Plots - Saved in {}".format(path_scatter_plots))

    # Analyse CSA - age
    logger.info("\nCSA and age:")
    path_model_age = os.path.join(path_model, 'age')
    if not os.path.exists(path_model_age):
        os.mkdir(path_model_age)
    lin_model = generate_linear_model(df['age'], y)
    save_model(lin_model, 'linear_fit', path_model_age )  # Linear model
    analyse_age(df['age'], y, path_model_age, lin_model)  # Quadratic model

    # Analyse CSA - sex
    logger.info("\nCSA and sex:")
    path_model_sex = os.path.join(path_model, 'sex')
    if not os.path.exists(path_model_sex):
        os.mkdir(path_model_sex)
    compare_sex(df, path_model_sex)  # T-Test and violin plots

    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05

    # Compute linear regression with all predictors and stepwise, compares, analyses and saves results
    logger.info("\nMultivariate model:\n")

    # Initialize dict for new COV of normalized CSA
    df_COV = {}

    # Create interaction variables
    x['inter BV_sex'] = x['brain volume']*x['sex']
    x['inter TV_sex'] = x['thalamus volume']*x['sex']

    # Loop through models
    for model, predictors in MODELS.items():
        logger.info("Initial predictors for {} are {}".format(model, predictors))
        if not os.path.exists(os.path.join(path_model, model)):
            os.mkdir(os.path.join(path_model, model))
        COV_step, COV_full = compute_regression_csa(x[predictors], y, p_in, p_out, "CSA_PMJ", os.path.join(path_model, model))
        df_COV[model] = [COV_step, COV_full]
    # Save as .csv COV of normalized CSA
    df_to_csv(pd.DataFrame.from_dict(df_COV, orient='index', columns=['Stepwise', 'Full']), os.path.join(path_model, 'norm_COV.csv'))

    # Check interaction sex and total brain volume and sex and thalamus volume
    results_BV = generate_linear_model(x[['brain volume', 'sex', 'inter BV_sex']], y)
    logger.info(results_BV.summary())
    results_TV = generate_linear_model(x[['thalamus volume', 'sex', 'inter TV_sex']], y)
    logger.info(results_TV.summary())

if __name__ == '__main__':
    main()
