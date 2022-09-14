import math
import random
import time

import numpy as np
from sklearn import svm, feature_selection, preprocessing
from scipy.stats import entropy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, precision_recall_curve
from sklearn.model_selection import GridSearchCV, LeavePOut, LeaveOneOut, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_probs(f, df, mid=None):
    # calculate p(x) for every x in f
    # if mid isn't none then the feature is continuous
    # and the split feature b the mid-point of the values range
    if mid is None:
        vals = df[f].value_counts().to_list()
        return [val / df.shape[0] for val in vals]
    else:
        b = len(df[df[f] <= mid].index) / df.shape[0]
        a = len(df[df[f] > mid].index) / df.shape[0]
        return [a, b]


def calculate_entropy(f, df, mid=None):
    return entropy(get_probs(f, df, mid), base=2)


def is_categorical(f, df):
    #  consider a feature categorical if it has no more than 10 unique values
    return df[f].nunique() < 10


def calc_entropy_known_y_categoical(x, y, df, mids):
    vals = df[y].value_counts().to_dict()
    probs = {key: val / df.shape[0] for key, val in vals.items()}
    hxy = 0
    for key, val in probs.items():
        dff = df[df[y] != key]
        hxy += val * calculate_entropy(x, dff, mids[x])
    return hxy


def calc_entropy_known_y_continiouos(x, y, df, mids):
    ab = get_probs(y, df, mids[y])
    hxy = 0
    dff = df[df[y] > mids[y]]
    hxy += ab[0] * calculate_entropy(x, dff, mids[x])
    dff = df[df[y] <= mids[y]]
    hxy += ab[1] * calculate_entropy(x, dff, mids[x])
    return hxy


def SU(x, y, entropies, df, mids):
    hx = entropies[x]
    hy = entropies[y]
    hxy = calc_entropy_known_y_categoical(x, y, df, mids) if mids[y] is None else calc_entropy_known_y_continiouos(x, y,
                                                                                                                   df,
                                                                                                                   mids)
    return 2 * (hx - hxy) / (hx + hy)


def get_mid(f, df):
    if is_categorical(f, df):
        return None
    return (df[f].max() + df[f].min()) / 2


def remove_irrelevant_features(features, sus):
    d = len(features)
    d_log_d = math.floor(d / math.log(d)) - 1
    sus_max = max(sus.values())
    sus_list = sorted(list(sus.values()))
    ro_0 = min(0.1 * sus_max, sus_list[d_log_d])
    return [f for f in features if sus[f] >= ro_0]


def find_weak_correlation_features(f1, U1, sus, d):
    d_log_d = d / math.log(d)
    dts = {fi: abs(sus[f1] - sus[fi]) for fi in U1}
    ro1 = max(dts.values()) * d_log_d
    return [fi for fi in U1 if dts[fi] <= ro1]


def find_similar_features(f1, U1, sus, df, entropies, mids):
    suf1 = {f: SU(f1, f, entropies, df, mids) for f in U1}
    ret = []
    for f in U1:
        mini = min(sus[f], sus[f1])
        if suf1[f] >= mini:
            ret.append(f)
    return ret


def FCFC_clustering(Ft, sus, df, entropies, mids):
    clusters = []
    U0 = sorted(Ft, key=lambda x: sus[x])
    d = len(U0)
    while len(U0) > 1:
        f1 = U0[0]
        U1 = U0.copy()[1:]
        c = [f1]
        U1 = find_weak_correlation_features(f1, U1, sus, d)
        c += find_similar_features(f1, U1, sus, df, entropies, mids)
        clusters.append(c)
        U0 = [ele for ele in U0 if ele not in c]
    if len(U0) != 0:
        clusters.append(U0)
    return clusters


def init_particles(clusters, sus):
    cvs = [max([sus[f] for f in c]) for c in clusters]
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


def IBPSO(clusters, sus, max_iters, df):
    x = init_particles(clusters, sus)
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
        print(gbest)
        i += 1
    ft = [c[j - 1] for c, j in zip(clusters, gbest) if j > 0]
    return ft


def HFS_C_P(X, Y, max_iters=50):
    df = pd.DataFrame(X)
    df['y'] = Y
    cols = df.columns.to_list()
    print(len(cols))
    features = cols.copy()
    features.remove('y')
    mids = {f: get_mid(f, df) for f in cols}
    entropies = {f: calculate_entropy(f, df, mids[f]) for f in cols}
    sus = {f: SU(f, 'y', entropies, df, mids) for f in features}
    Ft = remove_irrelevant_features(features, sus)
    print(len(Ft))
    print(Ft)
    clusters = FCFC_clustering(Ft, sus, df, entropies, mids)
    print(len(clusters))
    print(clusters)
    chosen = IBPSO(clusters, sus, max_iters, df)
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
    fit_times = []
    pred_times = []
    aucs = []
    accs = []
    mccs = []
    praucs = []
    total_folds = cv.get_n_splits(X, y)
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
            aucs.append(roc_auc_score(y_test, probs, multi_class='ovo'))
        except:
            try:
                aucs.append(roc_auc_score(y_test, probs[:, 1], multi_class='ovo'))
            except:
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
    for i in [2,6,9,10,19]:
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
            t = feature_selection.SelectKBest(HFS_C_P, k=k)
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
                row_auc = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds, 'AUC', aucs,
                           str(names_f), scores]
                save.loc[len(save)] = row_auc
                row_acc = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds, 'ACC', accs,
                           str(names_f), scores]
                save.loc[len(save)] = row_acc
                row_mcc = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds, 'MCC', mccs,
                           str(names_f), scores]
                save.loc[len(save)] = row_mcc
                row_prauc = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds, 'PR-AU',
                             praucs, str(names_f), scores]
                save.loc[len(save)] = row_prauc
                row_pred_time = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds,
                                 'prediction time', pred_times, str(names_f), scores]
                save.loc[len(save)] = row_pred_time
                row_fit_time = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds, 'fit time',
                                fit_times, str(names_f), scores]
                save.loc[len(save)] = row_fit_time
                row_selection_time = [names[i], X.shape[0], X.shape[1], 'HFS-C-P', m, k, cv_name, total_folds,
                                      'feature selection time', feature_selection_time, str(names_f), scores]
                save.loc[len(save)] = row_selection_time

        save.to_csv(f'res_{i}.csv')
