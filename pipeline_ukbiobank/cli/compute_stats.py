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
import matplotlib.pyplot as plt # pas utilisé à date
import statsmodels.api as sm
import seaborn as sns

from statsmodels.formula.api import ols # à retirer
from textwrap import dedent
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression # verify is used
from statsmodels.stats.multicomp import pairwise_tukeyhsd # verify if used

FNAME_LOG = 'log_stats.txt'

plt.style.use('seaborn') # pretty matplotlib plots

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

PREDICTORS = ['Sex', 'Height', 'Weight', 'Intracranial volume', 'Age'] # Add units of each

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

#3 Multivariate regression (stepwise)
    #normalise data before ???
    #full and stepwise model --> ok, ajouter AIC
    #output coeff.txt   --> ok for stepwise
#4 Test tuckey diff T1w et T2w
#5 Pertinence modèle
    # Résidus --> OKKKKKK
    # R^2 --> ok, faire un texte
    # Analyse variance
    # Intervalle de prédiction
#6 Logfile --> OK, mais en ajouter partout
# TODO: gérer l'écriture des fichiers

def compute_statistics(df):
    """
    Computes statistics such as mean, std, COV, etc. of CSA values per contrast type
    Args:
        df (panda.DataFrame): dataframe of all parameters and CSA values
    Returns:
        stats_df (panda.DataFrame): dataframe of statistics of CSA per contrat type
    """
    contrasts = ['T1w_CSA', 'T2w_CSA']
    metrics = ['number of sub','mean','std','med','95ci', 'COV', 'max', 'min', 'normality_test_p']
    stats = {}

    for contrast in contrasts:
        stats[contrast] = {}
        for metric in metrics:
            stats[contrast][metric] = {}

    for contrast in contrasts:
        stats[contrast]['number of sub'] = len(df[contrast])
        stats[contrast]['mean'] = np.mean(df[contrast])
        stats[contrast]['std'] = np.std(df[contrast])
        stats[contrast]['med']= np.median(df[contrast])
        stats[contrast]['95ci']= 1.96*np.std(df[contrast])/np.sqrt(len(df[contrast]))
        stats[contrast]['COV'] = np.std(df[contrast]) / np.mean(df[contrast])
        stats[contrast]['max'] = np.max(df[contrast])
        stats[contrast]['min'] = np.min(df[contrast])
        # Validate normality of data with Shapiro-wilik test
        stats[contrast]['normality_test_p'] = scipy.stats.shapiro(df[contrast])[1]
    # Convert dict to dataFrame
    stats_df = pd.DataFrame.from_dict(stats)
    return stats_df

def compute_predictors_statistic(df):
    """
    Compute statistics such as mean, min, max for each predictor
    Args:
        df (panda.dataFrame): dataframe of all parameters and CSA values
    Returns:
        stats_df (panda.dataFrame): dataFrame with statistics of parameters 

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
    stats['Sex'] = {}
    stats['Sex']['%_M'] = 100*(np.count_nonzero(df['Sex']) / len(df['Sex']))
    stats['Sex']['%_F'] = 100 - stats['Sex']['%_M']
    # Writes statistics of parameters into a texte
    output_text_stats(stats)
    stats_df =pd.DataFrame.from_dict(stats)

    return stats_df

def output_text_stats(stats):
    """
    Embed statistical results into sentences so they can easily be copy/pasted into a manuscript.
    """
    logger.info('Sex statistic:')
    logger.info('{} % of male  and {} % of female.'.format(stats['Sex']['%_M'],
                                                                             stats['Sex']['%_F']))
    logger.info('Height statistic:')
    logger.info('   Height between {} and {} cm, median height {} cm, mean height {} cm.'.format(stats['Height']['min'],
                                                                            stats['Height']['max'],
                                                                            stats['Height']['med'],
                                                                            stats['Height']['mean']))
    logger.info('Weight statistic:')
    logger.info('   Weight between {} and {} kg, median weight {} kg, mean weight {} kg.'.format(stats['Weight']['min'],
                                                                            stats['Weight']['max'],
                                                                            stats['Weight']['med'],
                                                                            stats['Weight']['mean']))
    logger.info('Age statistic:')
    logger.info('   Age between {} and {} y.o., median age {} y.o., mean age {} y.o..'.format(stats['Age']['min'],
                                                                            stats['Age']['max'],
                                                                            stats['Age']['med'],
                                                                            stats['Age']['mean']))
    logger.info('Intracranial Volume statistic:')
    logger.info('   Intracranial volume between {} and {} mm^3, median intracranial volume {} mm^3, mean intracranial volume {} mm^3.'.format(stats['Intracranial volume']['min'],
                                                                            stats['Intracranial volume']['max'],
                                                                            stats['Intracranial volume']['med'],
                                                                            stats['Intracranial volume']['mean']))
def config_table(corr_table, filename): # add units to table
    plt.figure(linewidth=2,
           tight_layout={'pad':1},
           figsize=(15,4)
          )
    rcolors = plt.cm.BuPu(np.full(len(corr_table.index), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(corr_table.columns), 0.1))
    table = plt.table(np.round((corr_table.values).astype(np.double), 4), 
            rowLabels = corr_table.index,
            colLabels = corr_table.columns,
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

def df_to_CSV(df, filename):
    df.to_csv(filename)
    logger.info('Created: ' + filename)

def save_table(df_table, filename):
    config_table(df_table, filename+'.png')
    df_to_CSV(df_table, filename+'.csv')

def get_correlation_table(df) :
    corr_table = df.corr()
    return corr_table

def get_predictors_rank(x,y):

    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(x, y_T1w)
    return selector.ranking_

def generate_linear_model(x, y, selected_features):
    x = x[selected_features]
    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()
    return results

def compute_stepwise(X,y, threshold_in, threshold_out): # TODO: add AIC cretaria
    """
    Performs backword and forward feature selection based on p-values 
    
    Args:
        X (panda.DataFrame): Candidate predictors
        y (panda.DataFrame): Candidate predictors with target
        threshold_in: include a feature if its p-value < threshold_in
        threshold_out: exclude a feature if its p-value > threshold_out

    Retruns:
        included: list of selected features
    
    """
    included = []
    while True:
        changed = False
        #Forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype = np.float64)
        #print('Excluded', excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        
        #print(best_pval)
        if best_pval < threshold_in:
            best_feature = excluded[new_pval.argmin()] # problème d'ordre ici --> OK!!
            included.append(best_feature)
            changed=True
            logger.info('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            #print('worst', worst_feature)
            included.remove(worst_feature)
            if verbose:
                logger.info('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def save_model(model, model_name, path_model_contrast):
    logger.info('Saving {} ...'.format(model_name))
    def save_summary():
        summary_path = os.path.join(path_model_contrast,'summary')
        if not os.path.exists(summary_path):
            os.mkdir(summary_path)
        summary_filename = os.path.join(summary_path, 'summary_'+ model_name +'.txt')
        with open(summary_filename, 'w') as fh:
            fh.write(model.summary(title = model_name ).as_text())
        logger.info('Created: ' + summary_filename)
    
    def save_coeff():
        coeff_path = os.path.join(path_model_contrast, 'coeff')
        if not os.path.exists(coeff_path):
            os.mkdir(coeff_path)
        coeff_filename = os.path.join(coeff_path ,'coeff_'+ model_name +'.csv')
        (model.params).to_csv(coeff_filename, header = None)
        logger.info('Created: ' + coeff_filename)

    save_coeff()
    save_summary()

def compute_regression_CSA(x,y, p_in, p_out, contrast, path_model):
    path_model_contrast = path_model +'/'+ contrast
    if not os.path.exists(path_model_contrast):
        os.mkdir(path_model_contrast)
    
    logger.info("Stepwise linear regression {}:".format(contrast))
    selected_features = compute_stepwise(x, y, p_in, p_out)
    logger.info('For'+ contrast+ 'selected feature are : {}'.format(selected_features))

    # Generates model with p-value stepwise linear regression
    model = generate_linear_model(x,y, selected_features)
    title_m1 = 'Stepwise linear regression of ' + contrast
    m1_name = 'stepwise_'+ contrast
    # Save summary of the model and the coefficients of the regression
    save_model(model, m1_name, path_model_contrast)

    # Generates liner regression with all predictors
    model_full = generate_linear_model(x,y, PREDICTORS)
    title_m2 = 'Full linear regression of ' + contrast
    m2_name = 'fullLin_'+ contrast
    # Save summary of the model and the coefficients of the regression
    save_model(model_full, m2_name, path_model_contrast)
    
    # Compares full and reduced models
    compared_models = compare_models(model, model_full, m1_name, m2_name)
    logger.info('Comparing models: {}'.format(compared_models))
    compared_models_filename = os.path.join(path_model_contrast,'compared_models.png')
    save_table(compared_models,compared_models_filename ) # Saves to .png ans .csv
    
    # Residual analysis
    analyse_residuals(model, m1_name, data = pd.concat([x, y], axis=1), path = os.path.join(path_model_contrast,'residuals'))

def compare_models(model_1, model_2,model_1_name, model_2_name  ):
    columns = ['Model', 'R^2', 'R^2_adj', 'F_p-value','F_value', 'AIC', 'df_res']
    table = pd.DataFrame(columns= columns)
    table['Model'] = [model_1_name, model_2_name]
    table['R^2'] = [model_1.rsquared, model_2.rsquared]
    table['R^2_adj'] = [model_1.rsquared_adj, model_2.rsquared_adj]
    table['F_p-value'] = [model_1.f_pvalue, model_2.f_pvalue]
    table['F_value'] = [model_1.fvalue, model_2.fvalue]
    table['AIC'] = [model_1.aic, model_2.aic]
    table['df_res'] = [model_1.df_resid, model_2.df_resid]
    table = table.set_index('Model')
    return table

def analyse_residuals(model, model_name, data, path):
    # Residual analysis
    residual = model.resid
    # Generate graph of QQ plot | Vaidate normality hypothesis of residual
    
    fig, axis = plt.subplots(1,2, figsize = (12,4))
    plt.autoscale(1)
    axis[0].title.set_text('Quantile-quantile plot of residuals')
    axis[1].title.set_text('Residuals vs Fitted')

    axis[0] = sm.qqplot(residual, line = '45', ax= axis[0] )

    model_fitted_y = model.fittedvalues
    model_residuals = model.resid
    axis[1]= sns.residplot(x=model_fitted_y, y=data.columns[-1], data = data,
                            lowess=True, 
                            scatter_kws={'alpha': 0.5}, 
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    
    axis[1].set_xlabel('Fitted values')
    axis[1].set_ylabel('Residuals')

    # annotations
    model_abs_resid = np.abs(model_residuals)
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        axis[1].annotate(i, 
                                xy=(model_fitted_y[i], 
                                    model_residuals[i]))
    #Return fig, save in other function
    fig.suptitle(' Residual analysis of '+ model_name, fontsize=16)
    plt.tight_layout()
    if not os.path.exists(path):
        os.mkdir(path)
    fname_fig = os.path.join(path +'/res_plots_' + model_name + '.png')
    plt.savefig(fname_fig) # put general variable 
    logger.info('Created: ' + fname_fig)

def remove_subjects(df, dict_exclude_subj):
    """
    Removes subjects from exclude list if given and all subjects that are missing any parameter.
    Writes in log the list of removed subjects.
    Args:
        df (panda.DataFrame): Dataframe with all subjects parameters and CSA values.
    Returns
        df_updated (panda.DataFrame): Dataframe without exluded subjects.
    """
    # Initalize list of subjects to exclude with all subjects missing a parameter value
    subjects_removed = df.loc[pd.isnull(df).any(1), :].index.values
    # Remove all subjects passed form the exclude list
    # TODO : retirer le sub-
    
    for sub in dict_exclude_subj:
        sub_id = int(sub[4:])
        df = df.drop(index = sub_id)
        subjects_removed = np.append(subjects_removed, sub_id)
    df_updated = df.dropna(0,how = 'any').reset_index(drop=True)
    logger.info("Subjects removed: {}".format(subjects_removed))
    return df_updated

def init_path_results():
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
        
    # Creates a panda dataFrame from datafile input arg .csv 
    df = (pd.read_csv(args.dataFile)).set_index('Subject')
    
    # Creates a dict with subjects to exclude if input yml config file is passed
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
#_______________________________________________________________________________________________________ 
# Removes all subjects that are missing a parameter or CSA value and subjects from exclude list.
    df = remove_subjects(df, dict_exclude_subj) 

# Initializes path of statistical results 
    path_metrics, path_model = init_path_results()
# _____________________________________________________________________________________________________
# Compute stats

    #1.1 Compute stats of T1w CSA and T2w CSA
    stats_csa = compute_statistics(df)
    # Format and save CSA stats as a .png and .csv file
    metric_csa_filename = os.path.join(path_metrics, 'stats_csa.png') # retirer .png
    save_table(stats_csa, metric_csa_filename)
        
    #1.2. Compute stats of the predictors
    stats_predictors = compute_predictors_statistic(df)
    # Format and save stats of csa as a table png and .csv
    stats_predictors_filename = os.path.join(path_metrics, 'stats_param.png') # retirer.png
    save_table(stats_predictors, stats_predictors_filename)   
#_______________________________________________________________________________________________________ 
# Correlation matrix
    corr_table = get_correlation_table(df)
    logger.info("Correlation matrix: {}".format(corr_table))
    corr_filename = os.path.join(path_metrics,'corr_table.png')
    # Saves an .png and .csv file of the correlation matrix in the results folder
    save_table(corr_table, corr_filename)
#________________________________________________________________________________________________________    
# Stepwise linear regression
    x = df.drop(columns = ['T1w_CSA', 'T2w_CSA'])
    y_T1w = df['T1w_CSA']
    y_T2w = df['T2w_CSA']
    # P_values for forward and backward stepwise
    p_in = 0.25 # To check
    p_out = 0.25 # To check
    # Computes linear regression with all predictors and stepwise, compares, analyses and saves results
    compute_regression_CSA(x, y_T1w, p_in, p_out, "T1w_CSA", path_model)
    compute_regression_CSA(x, y_T2w, p_in, p_out, 'T2w_CSA', path_model)

if __name__ == '__main__':
    main()
