import time

import numpy as np
import math
import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, LeavePOut, KFold, LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

MIN_CHANGE = 0.01

'''
This function take a dataframe:df and subset of features:S 
and returns ranking of the features in an ordered list.
'''


def evaluate_subset(train_x, train_y, test_x, test_y, model):
    model = clone(model)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    return sklearn.metrics.accuracy_score(test_y, y_pred)


def find_missing_feature(all_features, ranked_features):
    for feature in all_features:
        if feature not in ranked_features:
            return feature


def SFS(train_x, train_y, test_x, test_y, model):
    model = clone(model)
    if len(train_x.columns) == 1:
        return [train_x.columns[0]]
    sets = [(train_x.drop(train_x.columns[[j]], axis=1), test_x.drop(test_x.columns[[j]], axis=1)) for j in
            range(len(train_x.columns))]
    sets_scores = [evaluate_subset(train_set, train_y, test_set, test_y, model) for train_set, test_set in sets]
    best_train_set, best_test_set = sets[np.argmax(sets_scores)]
    ranked_features = SFS(best_train_set, train_y, best_test_set, test_y, model)
    missing_feature = find_missing_feature(train_x.columns, ranked_features)
    ranked_features.append(missing_feature)
    return ranked_features


def make_set(Fi, Fj):
    if Fi is None:
        return Fj
    if Fj is None:
        return Fi
    for f in Fj:
        if f not in Fi:
            Fi.append(f)
    return Fi


def evaluate_by_features(F, train_x, train_y, test_x, test_y, r, model):
    train_subset = train_x[F]
    test_subset = test_x[F]
    ranked_features = SFS(train_subset, train_y, test_subset, test_y, model)
    return find_top_r(train_subset, train_y, test_subset, test_y, ranked_features, r, model)


def find_bestSet(Fi, Fj, train_x, train_y, test_x, test_y, r, model):
    F = make_set(Fi, Fj)
    f_f, f_a = evaluate_by_features(F, train_x, train_y, test_x, test_y, r, model)
    fi_f, fi_a = evaluate_by_features(Fi, train_x, train_y, test_x, test_y, r, model)
    fj_f, fj_a = evaluate_by_features(Fj, train_x, train_y, test_x, test_y, r, model)
    A = [f_a, fi_a, fj_a]
    F = [f_f, fi_f, fj_f]
    return F[np.argmax(A)]


def find_set_alpha(f_best, f_curr, train_x, train_y, test_x, test_y, r, model):
    Fbb = make_set(f_best, f_curr)
    return evaluate_by_features(Fbb, train_x, train_y, test_x, test_y, r, model)


def find_top_r(train_x, train_y, test_x, test_y, ranked_features, r, model):
    top_r = ranked_features[:r]
    train_x = train_x[top_r]
    test_x = test_x[top_r]
    a = evaluate_subset(train_x, train_y, test_x, test_y, model)
    return top_r, a


def block_reduction(df, h, r, model):
    delta = math.inf
    alpha_best = 0
    f_best = None
    tries = 0
    while delta > MIN_CHANGE and tries < 3:
        tries += 1
        train, test = train_test_split(df, test_size=0.4)

        train_x = train.drop(columns=['y'])
        train_y = train['y']
        test_x = test.drop(columns=['y'])
        test_y = test['y']
        max_subsets = len(train_x.columns) - (len(train_x.columns) % h)

        S = [np.array([i for i in range(h)]) + j for j in range(0, max_subsets, h)]
        F = []
        A = []

        for s in tqdm(S):
            train_subset = train_x.iloc[:, s]
            test_subset = test_x.iloc[:, s]
            ranked_features = SFS(train_subset, train_y, test_subset, test_y, model)
            f, a = find_top_r(train_subset, train_y, test_subset, test_y, ranked_features, r, model)
            F.append(f)
            A.append(a)

        print(max(A))
        print('F', F[np.argmax(A)])
        f_curr = find_bestSet(F[0], F[1], train_x, train_y, test_x, test_y, r, model)
        for j in tqdm(range(2, len(F))):
            f_curr = find_bestSet(f_curr, F[j], train_x, train_y, test_x, test_y, r, model)
        print(f'---------------------------- iretation: {tries}')
        print(f_curr)
        f_curr, alpha_curr = find_set_alpha(f_best, f_curr, train_x, train_y, test_x, test_y, r, model)
        print(alpha_curr)
        if alpha_curr > alpha_best:
            delta = abs(alpha_curr - alpha_best)
            f_best = f_curr
            alpha_best = alpha_curr
            print(delta)

    return f_best


def get_cv(df):
    n = df.shape[0]
    if n < 50:
        return LeavePOut(2), 'LeavePairOut'
    if n < 100:
        return LeaveOneOut(), 'LeaveOneOut'
    if n < 1000:
        return StratifiedKFold(n_splits=10), '10Folds'
    return StratifiedKFold(n_splits=5), '5Folds'


def run_classification(model, X, y):
    cv, cv_name = get_cv(X)
    print(cv_name)
    fit_times = []
    pred_times = []
    aucs = []
    accs = []
    mccs = []
    praucs = []
    total_folds = cv.get_n_splits(X, y)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        fit_time = time.time()
        model.fit(X_train, y_train)
        fit_times.append(time.time() - fit_time)
        pred_time = time.time()
        probs = model.predict_proba(X_test)
        pred_times.append(time.time() - pred_time)
        preds = model.predict(X_test)
        try:
            aucs.append(roc_auc_score(y_test, probs, multi_class='ovo'))
        except Exception as e:
            try:
                aucs.append(roc_auc_score(y_test, probs[:, 1], multi_class='ovo'))
            except Exception as e:
                aucs.append(None)
        accs.append(accuracy_score(y_test, preds))
        mccs.append(matthews_corrcoef(y_test, preds))
        try:
            precision, _, _ = precision_recall_curve(y_test, probs[:, 1], pos_label=y_test.unique()[1,])
        except:
            precision = None
        praucs.append(precision)
    return fit_times, pred_times, aucs, accs, mccs, praucs, total_folds, cv_name


def get_model(model):
    models = {'RandomForest': RandomForestClassifier(), 'LogisticRegression': LogisticRegression(),
              'svm': SVC(probability=True),
              'NB': GaussianNB(), 'knn': KNeighborsClassifier()}
    return models[model]


def main():
    ks = [1, 2, 3, 4, 5, 10]
    names = ['CLL-SUB-111', 'ALLAML', 'BASEHOCK', 'COIL20', 'Carcinom', 'pone.0202167.s016', 'pone.0202167.s017',
             'bladderbatch', 'ayeastCC', 'breastCancerVDX', 'curatedOvarianData', 'leukemiasEset', 'Lung', 'Lymphoma',
             'MLL', 'SRBCT', 'CNS', 'pone.0202167.s011', 'pone.0202167.s012', 'pone.0202167.s015']



    for i in range(2, 3):
        try:
            save = pd.DataFrame(
                columns=['Dataset Name', 'Number of samples', 'Original Number of features', 'Filtering Algorithm',
                         'Learning algorithm', 'Number of features selected (K)', 'CV Method', 'Fold', 'Measure Type',
                         'Measure Value', 'List of Selected Features Names', 'Selected Features scores'])
            df = pd.read_csv(f'datasets/dataset_{i}.csv').drop(columns=['Unnamed: 0'])
            x = df.drop(columns=['y'])
            y = df['y']
            feature_selection_time = time.time()
            model = KNeighborsClassifier(n_neighbors=3)
            feature_names_in_ = block_reduction(df, 10, 10, model)
            feature_selection_time = time.time() - feature_selection_time
            for k in ks:
                feature_selection_time_k = time.time()
                kbest = SelectKBest(k=k)
                kbest.fit(x[feature_names_in_], y)
                feature_selection_time = time.time() - feature_selection_time_k + feature_selection_time
                names_f = kbest.get_feature_names_out()
                X = x[names_f]
                models = ['svm', 'knn', 'RandomForest', 'NB', 'LogisticRegression']
                for m in models:
                    model = get_model(m)
                    ind = [np.where(kbest.feature_names_in_ == name)[0][0] for name in names_f]
                    scores = kbest.scores_[ind]
                    fit_times, pred_times, aucs, accs, mccs, praucs, total_folds, cv_name = run_classification(model,
                                                                                                               pd.DataFrame(
                                                                                                                   X), y)
                    row_auc = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds, 'AUC', aucs,
                               str(names_f), scores]
                    save.loc[len(save)] = row_auc
                    row_acc = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds, 'ACC', accs,
                               str(names_f), scores]
                    save.loc[len(save)] = row_acc
                    row_mcc = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds, 'MCC', mccs,
                               str(names_f), scores]
                    save.loc[len(save)] = row_mcc
                    row_prauc = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds, 'PR-AU',
                                 praucs, str(names_f), scores]
                    save.loc[len(save)] = row_prauc
                    row_pred_time = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds,
                                     'prediction time', pred_times, str(names_f), scores]
                    save.loc[len(save)] = row_pred_time
                    row_fit_time = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds, 'fit time',
                                    fit_times, str(names_f), scores]
                    save.loc[len(save)] = row_fit_time
                    row_selection_time = [names[i], X.shape[0], X.shape[1], 'top-R', m, k, cv_name, total_folds,
                                          'feature selection time', feature_selection_time, str(names_f), scores]
                    save.loc[len(save)] = row_selection_time
            print('-----------------------------------------')
            print(i)
            save.to_csv(f'top_r/top_r{i}.csv')
        except Exception as e:
            print(e)
            print(f'WORNING - {i} failed')


def main2():
    cols = ['y'] + [f'x{i}' for i in range(44)]
    df = pd.read_csv('spect_SPECTF-train.csv', header=0, names=cols)
    model = KNeighborsClassifier(n_neighbors=3)
    feature_names_in_ = block_reduction(df, 10, 10, model)

main2()
