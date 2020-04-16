from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def log_cross_val_auc(clf, X, y, cv_splits, log_prefix):
    cv_auc = cross_val_score(clf, X, y, cv=cv_splits, scoring='roc_auc')
    cv_auc_mean = cv_auc.mean()
    cv_auc_error = cv_auc.std() * 2
    log = "{}_auc_cv: {:.5f} (+/- {:.5f})"
    print(log.format(log_prefix, cv_auc_mean, cv_auc_error))


def log_auc(clf, X, y, log_prefix):
    y_pred_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_pred_proba[:, 1])
    log = '{}_auc: {:.5f}'
    print(log.format(log_prefix, auc))


def train_pipeline(pipeline, X, y, cv_splits):
    # fit pipeline to cross validation splits
    if cv_splits > 1:
        log_cross_val_auc(pipeline, X, y, cv_splits, 'train')
    # fit pipeline to all training data
    pipeline.fit(X, y)
    log_auc(pipeline, X, y, 'train')
    return pipeline


def test_pipeline(pipeline, X, y):
    log_auc(pipeline, X, y, 'test')
