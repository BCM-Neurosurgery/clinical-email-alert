import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay   
import scipy.stats as stats
import statsmodels.api as sm

OLD_PAPER_PTS = {'B001': (-48, 100), 'B002': (-6, 296), 'B004': (-9, 803), 'B005': (-44, 578), 'B006': (-13, 585), 'B007': (-13, -1), 'B008': (-19, 141), 'B009': (1106, 1224), 'B010': (-51, -29), 'U001': (-25, 43), 'U003': (-20, 14)}

# Adjust data for delta model
def delta_model(df, features, pts):
    for id in pts:
        preDBS = df.query('pt_id == @id and days_since_dbs < 0')
        for f in features:
            df.loc[df['pt_id'] == id, f] -= preDBS[f].mean()
         
    return df

def run_regression(df: pd.DataFrame, pts: list, delta: bool, features: list, 
                   val_type: str, num_test: float = None, plot: bool = False, 
                   ax: plt.Axes = None, label: str = None, plot_thresh: bool = False, 
                   verbose: bool = False):
    """
    Main regression function to train and test data for the burdened state classification. Three different validation methods
    can be used, leave-one-patient-out (LOPO), Provenza et al. (2024) train and new data validation (VAL), or random (RAND).
    If the random validation is used, a percentage of the data to test on must be specified. The model can also be ran using
    the daily model or delta model which normalizes all patients to their pre-DBS mean of each feature. If the delta model is
    used, all patients specified in the list must have some data in the pre-DBS and post-DBS zone or an error will be thrown.

    args:
        df (pd.DataFrame): Main dataframe with all the data with a column, 'pt_id' specifying the patient id, a 'State Label'
                           feature column, a 'days_since_dbs' column, and columns for all specified features.
        pts (list): A list of all patient id strings to be analyzed.
        delta (bool): Boolean to specify using the delta model or not
        features (list): A list of features to run the regression on.
        val_type (str): A string of the validation type the model uses (LOPO, VAL, RAND).
        num_test (float): Float specifying the percentage of data to test on. Should only be stated if random validation is used.
        plot (bool): Whether or not to plot the AUROC curve of the model run.
        ax (plt.Axes): Axes object to plot the AUROC curve to, intialized as None.
        label (str): Label to associate with AUROC curve and display on the plot legend.
        plot_thresh(bool): Plot each patient's feature over time with regression threshold overlaid. Works best when model is
                           trained on one feature. Only works with LOPO regression.
        verbose (bool): Whether or not to print model statistics, accuracy, and the confusion matrix.

    returns:
        roc_auc (float): ROC AUC score for the model run.
        fpr (ndarray): Array of the false positive rate for the model run.
        tpr (ndarray): Array of the true positive rate for the model run.
        acc (float): Balanced accuracy score for the model run.
        cm (ndarray): Array of the confusion matrix for the model run.
    """
    
    labels = ['State_Label', 'pt_id', 'days_since_dbs']
    df = df.query('pt_id in @pts')
    if delta:
        df = delta_model(df, features, pts) # adjust feature values using delta model
    df = df[features + labels]
    df = df.dropna(subset=features)

    for f in features:
        df = df[np.abs(stats.zscore(df[f])) < 5]
    
    true_label = []
    pred = []
    pred_label = []
    
    test_y = []
    train_y = []

    if val_type == 'LOPO':
        if plot_thresh:
            fig, axs = plt.subplots(nrows=np.ceil(len(pts) / 3).astype(int), ncols=3, figsize=(20,np.ceil(len(pts) / 3).astype(int)*3), sharey=True)
        for i, pt in enumerate(pts):
            test_data = df.query('pt_id == @pt')
            train_data = df.query('pt_id != @pt')

            test_y = test_data['State_Label']
            train_y = train_data['State_Label']

            test_X = test_data[features]
            train_X = train_data[features]

            model = LogisticRegression(class_weight='balanced', penalty=None).fit(train_X, train_y)

            true_label.extend(test_y)
            pred.extend(model.predict_proba(test_X)[:,1])
            pred_label.extend(model.predict(test_X))

            if plot_thresh:
                axs.flatten()[i] = plot_regression_threshold(axs.flatten()[i], pt, df, model, features)
                fig.tight_layout(pad=3.0)

    elif val_type == 'VAL':
        test_data = pd.DataFrame()
        train_data = pd.DataFrame()
        for pt in pts:
            if pt in OLD_PAPER_PTS.keys():
                (start, end) = OLD_PAPER_PTS[pt]
                test_data = pd.concat([test_data, df.query('pt_id == @pt and (days_since_dbs < @start or days_since_dbs > @end)')], ignore_index=True)
                train_data = pd.concat([train_data, df.query('pt_id == @pt and (days_since_dbs >= @start and days_since_dbs <= @end)')], ignore_index=True)
            else:
                test_data = pd.concat([test_data, df.query('pt_id == @pt')], ignore_index=True)

    elif val_type == 'RAND':
        if num_test < 0 or num_test > 1:
            print(f'{num_test} must be a fraction between 0 and 1.')
            return ValueError
        train_data = df.sample(frac=num_test)
        test_data = pd.concat([df, train_data]).drop_duplicates(keep=False)

    else:
        print(f'{val_type} is not a valid model type.')
        return ValueError
    
    if val_type != 'LOPO':
        test_y = test_data['State_Label']
        train_y = train_data['State_Label']

        test_X = test_data[features]
        train_X = train_data[features]

        model = LogisticRegression(class_weight='balanced', penalty=None).fit(train_X, train_y)

        true_label.extend(test_y)
        pred.extend(model.predict_proba(test_X)[:,1])
        pred_label.extend(model.predict(test_X))

    cm = confusion_matrix(true_label, pred_label)

    acc = balanced_accuracy_score(true_label, pred_label)

    if verbose:
        print(f'Confusion matrix: \n{cm}\n')
        print(f'Balanced accuracy = {acc}\n')
        logit_model = sm.Logit(train_y, train_X)
        result = logit_model.fit()
        print(result.summary())

    fpr, tpr, _ = roc_curve(true_label, pred)
    roc_auc = roc_auc_score(true_label, pred)

    if plot:
        plot_regression(roc_auc, fpr, tpr, ax, label)

    return roc_auc, fpr, tpr, acc, cm

def plot_regression(roc_auc, fpr, tpr, ax, title):
        plot = RocCurveDisplay(fpr=fpr,tpr=tpr, roc_auc=roc_auc, estimator_name=title)
        plot.plot(ax=ax)

        return ax

def plot_regression_threshold(ax, pt, df, model, features):
    colors = {0: 'gold', 1: 'blue'}
    for f in features:
        ax.scatter(df.query('pt_id == @pt')['days_since_dbs'], df.query('pt_id == @pt')[f], c=[colors[label] for label in df.query('pt_id == @pt')['State_Label']], s=0.5)
        ax.plot(df.query('pt_id == @pt')['days_since_dbs'], df.query('pt_id == @pt')[f].rolling(window=5, min_periods=1).mean(), color='lightgray', lw=2, alpha=1)
    ax.hlines(-model.intercept_[0] / model.coef_[0][0], *ax.get_xlim(), color='black', ls='--')
    if 0 > df.query('pt_id == @pt')['days_since_dbs'].values[0] and 0 < df.query('pt_id == @pt')['days_since_dbs'].values[-1]:
        ax.vlines(0, ymin=-1.5, ymax=1.5, color='hotpink', ls='--' ,lw=3)
    if not df.query('pt_id == @pt and State_Label == 1').empty and pt in ['B001', 'B004', 'B005', 'B011', 'B013']:
        f = features[0] if len(features) == 1 else features[1]
        day_cross = np.where(df.query('pt_id == @pt')[f].rolling(window=5, min_periods=1).mean() <= -model.intercept_[0] / model.coef_[0][0])[0][0]
        ax.vlines(day_cross, ymin=-1.5, ymax=1.5, color='red', ls='--', lw=2, label=f'{day_cross}')
        print(day_cross)
    ax.set(xlabel='Days since DBS', ylabel=f, title=f'{pt}', xlim=ax.get_xlim())
    return ax 