"""Nested K-folds paradigm."""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import f_classif, SelectPercentile

from clr_callback import CyclicLR
from keras.callbacks import EarlyStopping


def train_test(model_type, data, labels, feature_names, percentile=[20, 40, 60, 80]):
    """ Train/test model using nested k-folds paradigm.

    Parameters
    ----------
    model_type : string
        'SVM' for support vector machine or 'DNN' for neural network.
    data : np.array
        features to use for training/testing of shape (n_samples, n_features)
    labels : np.array or list
        labels used for binary classification, ex: np.array of 1 = cannabis
        or 0 = control
    percentile : list
        list of percentiles to use in feature selection during grid search
        allows accommodatation of varying number of features
    feature_names : list
        list of feature names

    Returns
    -------
    df_performance : pd.DataFrame
        data frame of hyperparameters and performance across folds
    all_tp : list
        list of the number of true positives in each fold
    all_tn : list
        list of the number of true negatives in each fold
    all_fp : list
        list of the number of false positives in each fold
    all_fn : list
        list of the number of false negatives in each fold
    top_features : list
        list of selected features following grid search of percentile
    y_test_plot : list
        list of test labels for each fold (for plotting performance)
    probas_plot : list
        list of prediction probabilites (for plotting performance)
    """
    warnings.filterwarnings("ignore")

    # Log performance
    if model_type == 'SVM':
        col_header = ['Kernel', 'Gamma', 'Cost', 'Percentile', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    elif model_type == 'DNN':
        col_header = ['Optimizer', 'Initializer', 'Decay', 'Batch Size', 'Activation 1', 'Activation 2',
                      'Percentile', 'Sensitivity', 'Specificity', 'PPV', 'NPV']

    df_performance = pd.DataFrame(columns=col_header)
    all_tp = []
    all_tn = []
    all_fp = []
    all_fn = []

    # Plotting
    tprs = []
    aucs = []
    y_test_plot = []
    probas_plot = []

    # Feature importance
    top_features = []

    # Define grid hyper-parameters
    if model_type == 'SVM':
        tuned_parameters = dict(
                                    anova__percentile = percentile,
                                    svc__kernel = ['rbf', 'sigmoid', 'poly'],
                                    svc__gamma = [2**g for g in range(-15, 4)],
                                    svc__C = [2**C for C in range(-5, 16)]
                                )
    elif model_type == "DNN":
        tuned_parameters = dict(
                                    anova__percentile = percentile,
                                    nn__optimizer = ['SGD', 'AdamW'],
                                    nn__init = ['glorot_normal', 'glorot_uniform'],
                                    nn__activation_1 = ['relu', 'sigmoid', 'tanh'],
                                    nn__activation_2 = ['relu', 'sigmoid', 'tanh'],
                                    nn__batch_size = [32, 64, 128, 256],
                                    nn__decay = [10.0**i for i in range(-10,-1) if i%2 == 1]
                                )

    # Cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    inner_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    loop = 1
    folds = []
    for train_indices, test_indices in kfold.split(data, labels):
        print(f'Fold {loop}')

        # Callbacks for neural net
        clr = CyclicLR(mode='triangular', base_lr=0.175, max_lr=0.9175, step_size=12)
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
                           baseline=None, restore_best_weights=True)

        # Inner performance lists
        TP = []
        TN = []
        FP = []
        FN = []

        # Split data
        X_train = [data[idx] for idx in train_indices]
        y_train = [labels[idx] for idx in train_indices]
        X_test = [data[idx] for idx in test_indices]
        y_test = [labels[idx] for idx in test_indices]

        # Apply mean and variance centering
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Pipe feature selection and classifier together
        if model_type == 'SVM':
            anova = SelectPercentile(f_classif)
            svc = SVC(class_weight='balanced',  probability=True)
            clf = Pipeline([('anova', anova), ('svc', svc)])
        elif model_type == 'DNN':
            anova = ANOVASelection()  # Modified SelectPercentile class
            nn = KerasClassifier(build_fn=create_model, epochs=1000, verbose=0)
            clf = Pipeline([('anova', anova), ('nn', nn)])

        # Train model
        clf = GridSearchCV(clf, tuned_parameters, scoring='balanced_accuracy', n_jobs=30, cv=inner_kfold)
        #clf = RandomizedSearchCV(clf, tuned_parameters, n_iter=30, scoring='balanced_accuracy',
        #                         n_jobs=30, cv=inner_kfold)

        if model_type == 'SVM':
            clf.fit(X_train, y_train)
        elif model_type == 'DNN':
            clf.fit(X_train, y_train, nn__callbacks=[clr, es])


         # Determine top features from feature selection
        selection = SelectPercentile(f_classif, percentile=clf.best_estimator_[0].percentile).fit(X_train, y_train)
        top_indices = selection.get_support(indices=True)
        selected_features = []
        for idx in top_indices:
            selected_features.append(feature_names[idx])
        top_features.append(selected_features)

        # Test model
        y_true, y_pred = y_test, clf.predict(X_test)

        # Evaluate performance
        for idx, y in enumerate(y_true):
            if y == 1.0 and y == y_pred[idx]:
                TP.append(1)
            elif y == 1.0 and y != y_pred[idx]:
                FN.append(1)
            elif y == 0.0 and y == y_pred[idx]:
                TN.append(1)
            elif y == 0.0 and y != y_pred[idx]:
                FP.append(1)

        if len(FP) != 0 and len(FN) != 0:
            # This is most likely
            sensitivity = len(TP)/(len(TP)+len(FN))
            specificity = len(TN)/(len(TN)+len(FP))
            NPV = len(TN)/(len(TN)+len(FN))
            PPV = len(TP)/(len(TP)+len(FP))
        elif len(FP) != 0 and len(FN) == 0:
            # Likely overfitting
            sensitivity = 1
            specificity = len(TN)/(len(TN)+len(FP))
            NPV = 1
            PPV = len(TP)/(len(TP)+len(FP))
        elif len(FP) == 0 and len(FN) != 0:
            # Likely overfitting
            sensitivity = len(TP)/(len(TP)+len(FN))
            specificity = 1
            PPV = 1
            NPV = len(TN)/(len(TN)+len(FN))
        if len(FP) == 0 and len(FN) == 0:
            # Perfect classification - yeah right...
            sensitivity = 1
            specificity = 1
            NPV = 1
            PPV = 1

        all_tp.append(len(TP))
        all_tn.append(len(TN))
        all_fp.append(len(FP))
        all_fn.append(len(FN))

        # Append to performance df
        df_row_to_add = []

        if model_type == 'SVM':
            params = ['svc__kernel', 'svc__gamma', 'svc__C', 'anova__percentile']
        elif model_type == 'DNN':
            params = ['nn__optimizer', 'nn__init', 'nn__decay', 'nn__batch_size',
                      'nn__activation_1', 'nn__activation_2', 'anova__percentile']

        for param in params:
            df_row_to_add.append(clf.best_params_[param])
        df_row_to_add.append(sensitivity)
        df_row_to_add.append(specificity)
        df_row_to_add.append(PPV)
        df_row_to_add.append(NPV)
        folds.append('Fold ' + str(loop))
        df_performance = df_performance.append(pd.Series(df_row_to_add, index=df_performance.columns),
                                               ignore_index=True)
        df_performance.index = folds

        # For plotting
        y_test_plot.append(y_test)
        probas_ = clf.predict_proba(X_test)
        probas_plot.append(probas_)
        loop += 1

    return df_performance, all_tp, all_tn, all_fp, all_fn, top_features, y_test_plot, probas_plot
