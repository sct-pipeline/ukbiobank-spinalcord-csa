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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
#import seaborn as sns

import pipeline_ukbiobank.cli.select_subjects as ss
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multicomp import pairwise_tukeyhsd

FNAME_LOG = 'log_stats.txt'
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

PREDICTORS = ['Sex', 'Height', 'Weight', 'Intracranial volume', 'Age']

def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes the statistical analysis for the ukbiobank project",#add list of metrics that will be computed
        prog=os.path.basename(__file__).strip('.py')
        )
    parser.add_argument('-path-results',
                        required=False,
                        metavar='<dir_path>',
                        help="Folder that includes the output csv file from get_subkects_info")
    parser.add_argument('-exclude',
                        required=False,
                        help=".yml list of subjects to exclude from statistical analysis.") #add format of the list

    return parser
FILENAME = 'data_ukbiobank.csv'

#0. remove subjects
#1. Caractérisation des données
    #Nombre de sujet --> OK
    #Moyenne --> OK
    #mediane --> OK
    #Variance --> OK
    #COV --> OK
    #Int confiance (après je crois) --> OK
    # % H vs F --> OK
    # Plage de données pour taille, volume, poids, age --> OK
    #OUTPUT tableau + CSV (AJOUTER)
#2. Coff corrélation --> OK
    #OUTPUT tableau + CSV 
#3 Multivariate regression (stepwise)
    #output coeff.txt  
#4 Test tuckey diff T1w et T2w
#5 Pertinence modèle
    #collinéarité (scatterplot avec age et height)
    #Résidus
    #R^2
    #Analyse variance
    #Intervalle de prédiction
#6 Logfile
# add exclude parameter
def compute_statistics(df):
    """
    Compute statistics such as mean, std, COV, etc. per contrast type
    :param df Pandas structure
    """
    contrasts = ['T1w_CSA', 'T2w_CSA']
    metrics = ['number of sub','mean','std','med','95ci', 'COV', 'max', 'min']
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
        stats[contrast]['95ci'] = 1.96*np.std(df[contrast])/np.sqrt(len(df[contrast]))
        stats[contrast]['COV'] = np.std(df[contrast]) / np.mean(df[contrast])
        stats[contrast]['max'] = np.max(df[contrast])
        stats[contrast]['min'] = np.min(df[contrast])
    return stats

def compute_predictors_statistic(df):
    """
    Compute statistics such as mean, min, max for each predictor
    :param df Pandas structure
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
    return stats

def config_table(corr_table, filename):
    plt.figure(linewidth=2,
           tight_layout={'pad':1},
           figsize=(15,4)
          )
    rcolors = plt.cm.BuPu(np.full(len(corr_table.index), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(corr_table.columns), 0.1))
    table = plt.table(np.round(corr_table.values, 4), 
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


def get_correlation_table(df) :
    corr_table = df.corr()
    return corr_table

def compute_regression(df):
    #model = ols("T1w_CSA ~ Sex + Age + Weight + Height ", data=df)
    #results = model.fit()
    #print(results.params)
    x = df.drop(columns = ['T1w_CSA', 'T2w_CSA'])
    #x = sm.add_constant(x)
    y_T1w = df['T1w_CSA']
    #model = sm.OLS(y_T1w, x)
    #results = model.fit()
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(x, y_T1w)
    print(selector.ranking_)
    #print(results.summary())
#def get_scatterplot(x):
    #sns.pairplot(x)
def compute_stepwise(X,y, threshold_in, threshold_out): # TODO: add AIC cretaria
    """
    Performs backword and forward feature selection based on p-values 
    
    Args:
        X: panda.DataFrame with the candidate predictors
        y: panda.DataFrame with the candidate predictors with target
        threshold_in: include a feature if its p-value < threshold_in
        threshold_out: exclude a feature if its p-value > threshold_out

    Retruns:
        selected_features: list of selected features
    
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
            print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

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
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def main():
    parser = get_parser()
    args = parser.parse_args()
    df = (ss.load_participant_data_file(args.path_results +'/' + FILENAME)).set_index('Subject')

#1. Compute stats
    #1.1 Compute stats of T1w CSA and T2w CSA
    stats_csa = compute_statistics(df)
    stats_csa_df = pd.DataFrame.from_dict(stats_csa)
    config_table(stats_csa_df,  args.path_results + '/stats_CSA.png' )
    #add save as csv file
    #1.2. Compute stats of the predictors
    stats_predictors = compute_predictors_statistic(df)
    #add save as csv file and .png figure
#2. Correlation matrix
    corr_table = get_correlation_table(df)
    # Saves an .png of the correlation matrix in the results folder
    path_figures = args.path_results + '/figures'

    if not os.path.isdir(path_figures):
        os.mkdir(args.path_results + '/figures')
    config_table(corr_table, path_figures+ '/corr_table.png')
#3 Stepwise linear regression
    compute_regression(df)
    x = df.drop(columns = ['T1w_CSA', 'T2w_CSA'])
    y_T1w = df['T1w_CSA']
    y_T2w = df['T2w_CSA']
    p_in = 0.15
    p_out = 0.15
   
   # print(x.shape, y_T1w.shape)
    included= compute_stepwise(x, y_T1w, p_in, p_out)
    print('Included',included)


if __name__ == '__main__':
    main()
