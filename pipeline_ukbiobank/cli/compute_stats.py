#!/usr/bin/env python
# -*- coding: utf-8
# Computes statistical analysis for ukbiobank project
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
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

FNAME_LOG = 'log_stats.txt'

plt.style.use('seaborn')

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

PREDICTORS = ['Sex', 'Height', 'Weight', 'Age', 'Vscale', 'Volume ventricular CSF', 'GM volume', 'WM volume', 'Total brain volume norm', 'Total brain volume', 'Volume of thalamus (L)', 'Volume of thalamus (R)'] # Add units of each


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
    parser.add_argument('-exclude',
                        metavar='<file>',
                        required=False,
                        help=
                        "R|Config yaml file listing subjects to exclude from statistical analysis.\n"
                        "Yaml file can be validated at this website: http://www.yamllint.com/.\n"
                        "Below is an example yaml file:\n"
                        + dedent(
                        """
                        - sub-1000032
                        - sub-1000498
                        """)
                        )
    return parser


def compute_statistics(df):
    """
    Computes statistics such as mean, std, COV, etc. of CSA values
    Args:
        df (panda.DataFrame): dataframe of all parameters and CSA values
    Returns:
        stats_df (panda.DataFrame): statistics of CSA per contrast type
    """
    contrast = 'T1w_CSA'
    metrics = ['n','mean','std','med','95ci', 'COV', 'max', 'min', 'normality_test_p']
    stats = {}
    stats[contrast] = {}
    for metric in metrics:
        stats[contrast][metric] = {}
    # Computes the metrics
    stats[contrast]['n'] = len(df[contrast])
    stats[contrast]['mean'] = np.mean(df[contrast])
    stats[contrast]['std'] = np.std(df[contrast])
    stats[contrast]['med']= np.median(df[contrast])
    stats[contrast]['95ci']= 1.96*np.std(df[contrast])/np.sqrt(len(df[contrast]))
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
    logger.info('Statistics of {}:'.format(contrast))
    logger.info('   There are {} subjects included in the analysis.'.format(stats[contrast]['n']))
    logger.info('   CSA values are between {:.6} and {:.6} mm^2'.format(stats[contrast]['min'], stats[contrast]['max']))
    logger.info('   Mean CSA is {:.6} mm^2, standard deviation CSA is of {:.6}, median value is of {:.6} mm^2.'.format(stats[contrast]['mean'], stats[contrast]['std'], stats[contrast]['med']))
    logger.info('   The COV is of {:.6} and 95 confidence interval is {:.6}.'.format(stats[contrast]['COV'], stats[contrast]['95ci']))
    logger.info('   The results of Shapiro-wilik test has a p-value of {:.6}.'.format(stats[contrast]['normality_test_p']))


def compute_predictors_statistic(df):
    """
    Computes statistics such as mean, min, max for each predictor and writes it in the log file
    Args:
        df (panda.DataFrame): Dataframe of all predictor and CSA values
    Returns:
        stats_df (panda.DataFrame): Statistics of each predictors 
    """
    stats={}
    metrics = ['min', 'max', 'med', 'mean']
    for predictor in PREDICTORS:
        stats[predictor] = {}
        for metric in metrics:
            stats[predictor][metric] = {}
    for predictor in PREDICTORS:
        stats[predictor]['min'] = np.min(df[predictor])
        stats[predictor]['max'] = np.max(df[predictor])
        stats[predictor]['med'] = np.median(df[predictor])
        stats[predictor]['mean'] = np.mean(df[predictor])
    # Computes male vs female ratio
    stats['Sex'] = {}
    stats['Sex']['%_M'] = 100*(np.count_nonzero(df['Sex']) / len(df['Sex']))
    stats['Sex']['%_F'] = 100 - stats['Sex']['%_M']
    # Writes statistics of predictor into a text
    # output_text_stats(stats) ---> TODO: uncomment when updated
    stats_df =pd.DataFrame.from_dict(stats) # Converts dict to dataFrame

    return stats_df


def output_text_stats(stats): # TODO : add other parameters
    """
    Embed statistical results of predictors into sentences so they can easily be copy/pasted into a manuscript.
    Args: 
        stats (dict): dictionnary with stats of the predictors
    """
    logger.info('Sex statistic:')
    logger.info('   {:.3} % of male  and {:.3} % of female.'.format(stats['Sex']['%_M'],
                                                                             stats['Sex']['%_F']))
    logger.info('Height statistic:')
    logger.info('   Height between {} and {} cm, median height {} cm, mean height {} cm.'.format(stats['Height']['min'],
                                                                            stats['Height']['max'],
                                                                            stats['Height']['med'],
                                                                            stats['Height']['mean']))
    logger.info('Weight statistic:')
    logger.info('   Weight between {:.6} and {:.6} kg, median weight {:.6} kg, mean weight {:.6} kg.'.format(stats['Weight']['min'],
                                                                            stats['Weight']['max'],
                                                                            stats['Weight']['med'],
                                                                            stats['Weight']['mean']))
    logger.info('Age statistic:')
    logger.info('   Age between {} and {} y.o., median age {} y.o., mean age {:.6} y.o..'.format(stats['Age']['min'],
                                                                            stats['Age']['max'],
                                                                            stats['Age']['med'],
                                                                            stats['Age']['mean']))
    

def config_table(table, filename): 
    """
    Configures table and saves a .png file of it.
    Args:
        table (panda.DataFrame): table to save as a .png.
        filename (str): name that the table will be saved as.
    """
    plt.figure(linewidth=2,
           tight_layout={'pad':1},
           figsize=(15,4)
          )
    rcolors = plt.cm.BuPu(np.full(len(table.index), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(table.columns), 0.1))
    table = plt.table(np.round((table.values).astype(np.double), 4), 
            rowLabels = table.index,
            colLabels = table.columns,
            rowLoc='center',
            loc = 'center',
            cellLoc = 'center',
            colColours = ccolors,
            rowColours = rcolors
            )

    table.scale(1, 1.5)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)
 
    plt.draw()
    fig = plt.gcf()
    plt.savefig(filename,
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            )
    logger.info('Created: ' + filename)


def scatter_plot(x,y, filename, path):
    """
    Generate and save a scatter plot of y and x.
    Args:
        x (panda.DataFrame): predictor
        y (panda.DataFrame): CSA values
    """
    plt.figure()
    sns.regplot(x=x, y=y, line_kws={"color": "crimson"})
    plt.ylabel('CSA (mm^2)')
    plt.xlabel(filename)
    plt.title('Scatter Plot - CSA /'+ filename) # TODO: change name
    plt.savefig(os.path.join(path,filename +'.png'))


def df_to_csv(df, filename):
    """
    Saves a Dataframe as a .csv file.
    Args:
        df (panda.DataFrame)
        filename (str): Name of the output .csv file.
    """
    df.to_csv(filename)
    logger.info('Created: ' + filename)


def save_table(df_table, filename):
    """
    Saves a dataframe as a .png  and a .csv file.
    Args:
        df (panda.DataFrame)
        filename (str): Name of the output file (without .csv or .png).
    """
    config_table(df_table, filename +'.png') # Saves as a .png
    df_to_csv(df_table, filename +'.csv') # Saves as a .csv


def get_correlation_table(df) :
    """
    Returns the correlation matrix of a DataFrame.
    Args:
        df (panda.DataFrame)
    Returns:
        corr_table (panda.DataFrame): correlation matrix of df
    """
    corr_table = df.corr(method='pearson')
    return corr_table


def compare_gender(df):
    """
    Calculate the T-test for the means of two independent samples of scores
    Args:

    """
    results_t_test = scipy.stats.ttest_ind(df[df['Sex'] == 0]['T1w_CSA'], df[df['Sex'] == 1]['T1w_CSA'])
    logger.info("T test  - Male/Female CSA : {}".format(results_t_test))
    return results_t_test

def generate_linear_model(x, y, selected_predictors):
    """
    Computes linear regresion with the selected predictors.
    Args:
        x (panda.DataFrame): Data of the predictors
        y (panda.DataFrame): Data of CSA
        selected_predictors (list): List of predictors to inlcude in the linear regression model.
    Returns:
        results (statsmodels.regression.linear_model.RegressionResults object): fitted linear model.
    """
    # Updates x to only have the data of the selected predictors.
    x = x[selected_predictors] 
    # Adds a columns of ones to the original DataFrame.
    x = sm.add_constant(x) 

    model = sm.OLS(y, x) # Computes linear regression
    results = model.fit() # Fits model
    return results


def compute_stepwise(x, y, threshold_in, threshold_out):
    """
    Performs backward and forward predictor selection based on p-values 
    
    Args:
        x (panda.DataFrame): Candidate predictors
        y (panda.DataFrame): Candidate predictors with target
        threshold_in: include a predictor if its p-value < threshold_in
        threshold_out: exclude a predictor if its p-value > threshold_out
        ** threshold_in > threshold_out
    Returns:
        included: list of selected predictor
    
    """
    included = [] # Initialize a list for inlcuded predictors in the model
    while True:
        changed = False
        #Forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype = np.float64)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min() 
        if best_pval < threshold_in:
            best_predictor = excluded[new_pval.argmin()] # Gets the predictor with the lowest p_value
            included.append(best_predictor) # Adds best predictor to included predictor list
            changed=True
            logger.info('Add  {:30} with p-value {:.6}'.format(best_predictor, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit() # Computes linear regression with included predictor
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # Gets the worst p-value of the model
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_predictor = included[pvalues.argmax()] # gets the predictor with worst p-value
            included.remove(worst_predictor) # Removes the worst predictor of included predictor list
            logger.info('Drop {:30} with p-value {:.6}'.format(worst_predictor, worst_pval))
        if not changed:
            break

    return included


def save_model(model, model_name, path_model_contrast):
    """
    Saves summary in .txt file and coeff in a .csv file.

    Args:
        model (statsmodels.regression.linear_model.RegressionResults object):  fited linear model.
        model_name (str): Name of the model
        path_model_contrast (str): Path of the result folder for this model and constrast  
    """
    logger.info('Saving {} ...'.format(model_name))
    def save_summary():
        summary_path = os.path.join(path_model_contrast,'summary')
        # Creates if doesn't exist a folder for the model summary
        if not os.path.exists(summary_path):
            os.mkdir(summary_path)
        summary_filename = os.path.join(summary_path, 'summary_'+ model_name +'.txt')
        # Saves the summary of the model in a .txt file
        with open(summary_filename, 'w') as fh:
            fh.write(model.summary(title = model_name ).as_text())
        logger.info('Created: ' + summary_filename)
    
    def save_coeff():
        coeff_path = os.path.join(path_model_contrast, 'coeff')
        # Creates a folder for results coeff of the model if doesn't exists
        if not os.path.exists(coeff_path):
            os.mkdir(coeff_path)
        coeff_filename = os.path.join(coeff_path ,'coeff_'+ model_name +'.csv')
        # Saves the coefficients of the model in .csv file
        (model.params).to_csv(coeff_filename, header = None)
        logger.info('Created: ' + coeff_filename)

    save_coeff()
    save_summary()


def compute_regression_csa(x, y, p_in, p_out, contrast, path_model):
    """
    Computes stepwise model and complete linear model of CSA. Saves both models, compares them and analyses residuals.
    Args:
        x (panda.DataFrame): Data of predictors
        y (panda.DataFrame): Data of CSA
        p_in (float): include a predictor if its p-value < p_in for stepwise ** p_in < p_out
        p_out (float): exclude a predictor if its p-value > p_out for stepwise
        contrast (str): Contrast of the image that CSA value was computed from
        path_model (str): Path of the result folder of the models
    """
    # Creates directory for results of CSA model for this contrast if doesn't exists
    path_model_contrast = os.path.join(path_model, contrast)
    if not os.path.exists(path_model_contrast):
        os.mkdir(path_model_contrast)
    
    # Computes stepwise linear regression with p_value
    logger.info("Stepwise linear regression {}:".format(contrast))
    selected_predictors = compute_stepwise(x, y, p_in, p_out)
    logger.info('For '+ contrast + ' selected predictors are : {}'.format(selected_predictors))

    # Generates model with selected predictors from stepwise
    model = generate_linear_model(x,y, selected_predictors)
    m1_name = 'stepwise_'+ contrast
    # Saves summary of the model and the coefficients of the regression
    save_model(model, m1_name, path_model_contrast)

    # Generates linear regression with all predictors
    model_full = generate_linear_model(x,y, PREDICTORS)
    m2_name = 'fullLin_'+ contrast
    # Saves summary of the model and the coefficients of the regression
    save_model(model_full, m2_name, path_model_contrast)
    
    # Compares full and reduced models with F_value, R^2,...
    compared_models = compare_models(model, model_full, m1_name, m2_name)
    logger.info('Comparing models: {}'.format(compared_models))
    compared_models_filename = os.path.join(path_model_contrast,'compared_models')
    save_table(compared_models,compared_models_filename ) # Saves to .png ans .csv
    
    # Residual analysis
    logger.info('Analysing residuals...')
    analyse_residuals(model, m1_name, data = pd.concat([x, y], axis=1), path = os.path.join(path_model_contrast,'residuals'))
    analyse_residuals(model_full, m2_name, data = pd.concat([x, y], axis=1), path = os.path.join(path_model_contrast,'residuals'))


def compare_models(model_1, model_2, model_1_name, model_2_name):
    """
    Creates a dataframe with R^2, R^2 adjusted, F p_value, F_value, AIC and df of residuals for both models
    Args:
        model_1 (statsmodels.regression.linear_model.RegressionResults object): First fitted model to compare
        model_2 (statsmodels.regression.linear_model.RegressionResults object): Second fitted model to compare
        model_1_name (str): Name of the first model
        model_2_name (str): Name of the second model
    Returns:
        table (panda.DataFrame) : Dataframe with results of both models
    """
    columns = ['Model', 'R^2', 'R^2_adj', 'F_p-value','F_value', 'AIC', 'df_res']
    # Initialize a dataframe
    table = pd.DataFrame(columns= columns)
    table['Model'] = [model_1_name, model_2_name]
    table['R^2'] = [model_1.rsquared, model_2.rsquared]
    table['R^2_adj'] = [model_1.rsquared_adj, model_2.rsquared_adj]
    table['F_p-value'] = [model_1.f_pvalue, model_2.f_pvalue]
    table['F_value'] = [model_1.fvalue, model_2.fvalue]
    table['AIC'] = [model_1.aic, model_2.aic]
    table['df_res'] = [model_1.df_resid, model_2.df_resid]
    table = table.set_index('Model') # Set index to model names
    return table


def analyse_residuals(model, model_name, data, path): # TODO: Add residuals as function of each parameter
    """
    Generates and saves Residuals vs Fitted values plot and QQ plot.
    Args:
        model (statsmodels.regression.linear_model.RegressionResults object): fitted model to analyse residuals
        model_name (str): Name of the model
        data (panda.DataFrame): Data of all predictors and CSA values.
        path (str): Path to folder of the results of the residuals analysis.
    """
    # Get the residuals from the fitted model
    residual = model.resid
    # Initialize a plot
    fig, axis = plt.subplots(1,2, figsize = (12,4))
    plt.autoscale(1)
    axis[0].title.set_text('Quantile-quantile plot of residuals')
    axis[1].title.set_text('Residuals vs Fitted')

    # Generate graph of QQ plot | Vaidate normality hypothesis of residual
    axis[0] = sm.qqplot(residual, line = 's', ax= axis[0] )
    # Residual vs fitted values plot
    model_fitted_y = model.fittedvalues
    # Compute plot
    axis[1]= sns.residplot(x=model_fitted_y, y=data.columns[-1], data = data,
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
        axis[1].annotate(i, 
                                xy=(model_fitted_y[i], 
                                    residual[i]))
    
    # Title of the plot
    fig.suptitle(' Residual analysis of '+ model_name, fontsize=16)
    plt.tight_layout()
    # Create path to folder of the results of the residuals analysis if doesn't exists.
    if not os.path.exists(path):
        os.mkdir(path)
    fname_fig = os.path.join(path +'/res_plots_' + model_name + '.png')
    plt.savefig(fname_fig) # save plot 
    logger.info('Created: ' + fname_fig)


def remove_subjects(df, dict_exclude_subj):
    """
    Removes subjects from exclude list if given and all subjects that are missing a parameter.
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
        if sub[-10:-7]=='T1w':
            sub_id = (sub[:-11])
            # Check if the subjects is in the dataframe
            if sub_id in df.index:
                df = df.drop(index = sub_id)
                subjects_removed = np.append(subjects_removed, sub_id) # add subject to excluded list
    df_updated = df.dropna(0,how = 'any').reset_index(drop=True) # Drops all subjects missing a parameter
    logger.info("{} Subjects removed : {}".format(len(subjects_removed),subjects_removed))
    return df_updated


def init_path_results():
    """
    Creates folders for stats results if does not exists.
    Returns:
        path_metric: path to folder of metrics
        path_model: path to folder of stats models
    """
    path_statistics = 'stats_results'
    if not os.path.exists(path_statistics):
        os.mkdir(path_statistics)
    path_metrics = os.path.join(path_statistics,'metrics')
    if not os.path.exists(path_metrics):
        os.mkdir(path_metrics)
    path_model =  os.path.join(path_statistics,'models')
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

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), FNAME_LOG))
    logging.root.addHandler(fh)

    # Remove all subjects that are missing a parameter or CSA value and subjects from exclude list.
    df = remove_subjects(df, dict_exclude_subj) 

    # Initialize path of statistical results 
    path_metrics, path_model = init_path_results()

    # Compute stats for T1w CSA
    stats_csa = compute_statistics(df)
    # Format and save CSA stats as a .png and .csv file
    metric_csa_filename = os.path.join(path_metrics, 'stats_csa')
    save_table(stats_csa, metric_csa_filename)
        
    # Compute stats of the predictors
    stats_predictors = compute_predictors_statistic(df)
    # Format and save stats of csa as a table png and .csv
    stats_predictors_filename = os.path.join(path_metrics, 'stats_param')
    save_table(stats_predictors, stats_predictors_filename)   

    # Correlation matrix
    corr_table = get_correlation_table(df)
    logger.info("Correlation matrix: {}".format(corr_table))
    corr_filename = os.path.join(path_metrics,'corr_table')
    # Saves an .png and .csv file of the correlation matrix in the results folder
    save_table(corr_table, corr_filename)


    # Stepwise linear regression and complete linear regression
    x = df.drop(columns = ['T1w_CSA']) # Initialize x to data of predictors
    y_T1w = df['T1w_CSA']

    # Genereate scatter plot for all predictors and CSA values
    path_scatter_plots = os.path.join(path_metrics,'scatter_plots')
    if not os.path.exists(path_scatter_plots):
        os.mkdir(path_scatter_plots)
    for column, data in x.iteritems():
        scatter_plot(data, y_T1w, column, path_scatter_plots)
    logger.info("Created Scatter Plots - Saved in {}".format(path_scatter_plots))

    # Analyse CSA - Age

    # Analyse CSA - Sex
    rsults_t_test_gender = compare_gender(df)
    # P_values for forward and backward stepwise
    p_in = 0.1 # (p_in > p_out)
    p_out = 0.05
    # Computes linear regression with all predictors and stepwise, compares, analyses and saves results
    compute_regression_csa(x, y_T1w, p_in, p_out, "T1w_CSA", path_model)

if __name__ == '__main__':
    main()
