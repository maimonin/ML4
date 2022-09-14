import math
import random
import time

import numpy as np
from sklearn import svm, feature_selection, preprocessing
from scipy.stats import entropy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, precision_recall_curve
from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def remove_irrelevant_features(features, X, Y):
    model = RandomForestClassifier()
    model.fit(X, Y)
    importance = model.feature_importances_
    importance_max = importance.max()
    importance_map = {f: i for f, i in zip(features, importance)}
    ro_0 = 0.1 * importance_max
    return [f for f, i in zip(features, importance) if i >= ro_0], importance_map


def find_weak_correlation_features(f1, U1, importance, d):
    d_log_d = d / math.log(d)
    dts = {fi: abs(importance[f1] - importance[fi]) for fi in U1}
    ro1 = max(dts.values()) * d_log_d
    return [fi for fi in U1 if dts[fi] <= ro1]


def find_similar_features(f1, U1, importance, df):
    model = RandomForestRegressor()
    model.fit(df[U1], df[f1])
    importance_f1 = model.feature_importances_
    importance_f1 = {f: i for f, i in zip(U1, importance_f1)}
    ret = []
    for f in U1:
        mini = min(importance[f], importance[f1])
        if importance_f1[f] >= mini:
            ret.append(f)
    return ret


def FCFC_clustering(Ft, importance, df):
    clusters = []
    U0 = sorted(Ft, key=lambda x: importance[x])
    d = len(U0)
    while len(U0) > 1:
        f1 = U0[0]
        U1 = U0.copy()[1:]
        c = [f1]
        U1 = find_weak_correlation_features(f1, U1, importance, d)
        c += find_similar_features(f1, U1, importance, df)
        clusters.append(c)
        U0 = [ele for ele in U0 if ele not in c]
    if len(U0) != 0:
        clusters.append(U0)
    return clusters


def init_particles(clusters, importance):
    cvs = [max([importance[f] for f in c]) for c in clusters]
    max_cv = max(cvs)
    pcvs = [cv / max_cv for cv in cvs]
    x = []
    for i in range(len(clusters)):
        xi = []
        for j in range(len(clusters)):
            if random.uniform(0, 1) < pcvs[j]:
                xi.append(random.randint(1, len(clusters[j])))
            else:
                xi.append(0)
        x.append(xi)
    return x


def get_best(x, clusters, df, pbest=None, pbest_scores=None):
    k = 0
    new_pbests = []
    new_pbests_scores = []
    gbest = None
    gbest_score = None
    for xi in x:
        model = svm.SVC()
        fs = [c[xij - 1] for xij, c in zip(xi, clusters) if xij > 0]
        fx = df[fs]
        if fx.empty:
            score = 0
        else:
            y = df['y']
            model.fit(fx, y)
            score = model.score(fx, y)
        if pbest is None or pbest_scores[k] < score:
            new_pbests.append(xi)
            new_pbests_scores.append(score)
        else:
            new_pbests.append(pbest[k])
            new_pbests_scores.append(pbest_scores[k])
        if gbest is None or gbest_score < new_pbests_scores[-1]:
            gbest = new_pbests[-1]
            gbest_score = new_pbests_scores[-1]
        k += 1
    return new_pbests, new_pbests_scores, gbest


def IBPSO(clusters, importance, max_iters, df):
    x = init_particles(clusters, importance)
    pbest, pbest_scores = None, None
    gbest = None
    i = 0
    while gbest is None or max_iters > i:
        pbest, pbest_scores, gbest = get_best(x, clusters, df, pbest, pbest_scores)
        for pb, xi in zip(pbest, x):
            for j in range(len(xi)):
                if random.uniform(0, 1) > 0.5:
                    g = random.gauss(0, 1)
                    xi[j] = math.ceil((pb[j] + gbest[j]) / 2) + math.ceil(g * abs(pb[j] - gbest[j]))
                    if xi[j] < 0:
                        xi[j] = 0
                    elif xi[j] > len(clusters[j]):
                        xi[j] = len(clusters[j])
                else:
                    xi[j] = pb[j]
        i += 1
    ft = [c[j - 1] for c, j in zip(clusters, gbest) if j > 0]
    return ft


def New_HFS_C_P(X, Y, max_iters=50):
    df = pd.DataFrame(X)
    df['y'] = Y
    x = pd.DataFrame(X)
    cols = df.columns.to_list()
    features = x.columns.to_list()
    Ft, importance = remove_irrelevant_features(features, x, Y)
    clusters = FCFC_clustering(Ft, importance, df)
    chosen = IBPSO(clusters, importance, max_iters, df)
    ret = []
    for f in features:
        a = 1 if f in chosen else 0
        ret.append(a)
    return ret, None


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
    if len(y.value_counts()) > 2:
        mult = 'ovo'
    else:
        mult = 'raise'
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fit_time = time.time()
        model.fit(X_train, y_train)
        fit_times.append(time.time() - fit_time)
        pred_time = time.time()
        probs = model.predict_proba(X_test)
        pred_times.append(time.time() - pred_time)
        preds = model.predict(X_test)
        try:
            aucs.append(roc_auc_score(y_test, probs, multi_class=mult))
        except Exception as e:
            try:
                aucs.append(roc_auc_score(y_test, probs[:, 1], multi_class=mult))
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


if __name__ == "__main__":
    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
    models = ['svm', 'knn', 'RandomForest', 'NB', 'LogisticRegression']
    # ks = [1, 2]
    names = ['CLL-SUB-111', 'ALLAML', 'BASEHOCK', 'COIL20', 'Carcinom', 'pone.0202167.s016', 'pone.0202167.s017',
             'bladderbatch', 'ayeastCC', 'breastCancerVDX', 'curatedOvarianData', 'leukemiasEset', 'Lung', 'Lymphoma',
             'MLL', 'SRBCT', 'CNS', 'pone.0202167.s011', 'pone.0202167.s012', 'pone.0202167.s015']
    for i in [5]:
        print(f'{i} --------------------------------------------------------------------------------------------')
        df1 = pd.read_csv(f'datasets/dataset_{i}.csv').drop(columns=['Unnamed: 0'])
        X = df1.drop(columns=['y'])
        Y = df1['y']
        save = pd.DataFrame(
            columns=['Dataset Name', 'Number of samples', 'Original Number of features', 'Filtering Algorithm',
                     'Learning algorithm', 'Number of features selected (K)', 'CV Method', 'Fold', 'Measure Type',
                     'Measure Value', 'List of Selected Features Names', 'Selected Features scores'])
        for k in ks:
            start = time.time()
            t = feature_selection.SelectKBest(New_HFS_C_P, k=k)
            new_x = t.fit_transform(X, Y)
            feature_selection_time = time.time() - start
            names_f = t.get_feature_names_out()
            ind = [np.where(t.feature_names_in_ == name)[0][0] for name in names_f]
            scores = t.scores_[ind]
            for m in models:
                model = get_model(m)
                fit_times, pred_times, aucs, accs, mccs, praucs, total_folds, cv_name = run_classification(model,
                                                                                                           pd.DataFrame(
                                                                                                               new_x),
                                                                                                           Y)
                row_auc = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds, 'AUC', aucs,
                           str(names_f), scores]
                save.loc[len(save)] = row_auc
                row_acc = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds, 'ACC', accs,
                           str(names_f), scores]
                save.loc[len(save)] = row_acc
                row_mcc = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds, 'MCC', mccs,
                           str(names_f), scores]
                save.loc[len(save)] = row_mcc
                row_prauc = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds, 'PR-AU',
                             praucs, str(names_f), scores]
                save.loc[len(save)] = row_prauc
                row_pred_time = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds,
                                 'prediction time', pred_times, str(names_f), scores]
                save.loc[len(save)] = row_pred_time
                row_fit_time = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds, 'fit time',
                                fit_times, str(names_f), scores]
                save.loc[len(save)] = row_fit_time
                row_selection_time = [names[i], X.shape[0], X.shape[1], 'New_HFS-C-P', m, k, cv_name, total_folds,
                                      'feature selection time', feature_selection_time, str(names_f), scores]
                save.loc[len(save)] = row_selection_time

        # save.to_csv(f'res_{i}_new.csv')
