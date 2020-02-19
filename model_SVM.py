import tempfile
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import f_classif, SelectPercentile
from joblib import Memory
import warnings
warnings.filterwarnings("ignore")

# Read in covariate adjusted features (performed in R)
base_dir = '/mnt/Filbey/Ryan/currentProjects/SVM/Final/Combined/features'

# Diffusion data, thresholded at 0.3 during TBSS
data = np.genfromtxt(str(base_dir)+'/res.dmri_03.csv', delimiter=',')
labels = np.genfromtxt(str(base_dir)+'/group_dmri.csv', delimiter=',')

# Read in IDs
with open(str(base_dir)+'/ids_dmri.txt') as f:
    ids_strip = [id for id in f]
ids = [id.rstrip() for id in ids_strip]

# Read in feature names
with open(base_dir + '/feature_names.txt') as f:
    feature_names = [line.strip('\n') for line in f]

# Reshape labels
labels = np.reshape(labels, (319, 1))

# Lists for recording performance.
mean_sensitivity = []
mean_specificity = []
mean_PPV = []
mean_NPV = []
TP_ids = []
TN_ids = []
FP_ids = []
FN_ids = []
tprs = []
aucs = []
X_test_list = []
y_test_list = []
probas_list = []
best_parameters = []
best_features = []

# Define SVM Param Grid
cost = []
for c in range(-5, 16):
    cost.append(2**c)
gamma = []
for g in range(-15, 4):
    gamma.append(2**g)

tuned_parameters = dict(
                            anova__percentile = [20, 40, 60, 80],
                            svc__kernel = ['rbf', 'sigmoid', 'poly'],
                            svc__gamma = gamma,
                            svc__C = cost
                        )

# Cross-validation
outer_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
inner_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
for train_indices, test_indices in outer_kfold.split(data, labels):
    # Split data
    X_train = [data[idx] for idx in train_indices]
    y_train = [labels[idx] for idx in train_indices]
    X_test = [data[idx] for idx in test_indices]
    y_test = [labels[idx] for idx in test_indices]
    y_test_id = [ids[idx] for idx in test_indices]

    # Apply mean and variance noramlization
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Memory handling
    cachedir = tempfile.mkdtemp()
    mem = Memory(location=cachedir, verbose=0)
    f_classif = mem.cache(f_classif)

    # Pipe feature selection and classifier together
    anova = SelectPercentile(f_classif)
    svc = SVC(class_weight='balanced',  probability=True)
    clf = Pipeline([('anova', anova), ('svc', svc)])

    # Train model
    clf = GridSearchCV(clf, tuned_parameters, scoring='balanced_accuracy', n_jobs=-1, cv=inner_kfold)
    clf.fit(X_train, y_train)
    best_parameters.append(clf.best_params_)

    # Determine top features from feature selection
    selection = SelectPercentile(f_classif, percentile=clf.best_estimator_[0].percentile).fit(X_train, y_train)
    best_indices = selection.get_support(indices=True)
    selected_features = [feature_names[idx] for idx in best_indices]
    best_features.append(selected_features)

    # Test model
    y_true, y_pred = y_test, clf.predict(X_test)

    # Evaluate performance within CV fold
    TP = []
    TN = []
    FP = []
    FN = []
    for idx, y in enumerate(y_true):
        if y == 1.0 and y == y_pred[idx]:
            TP.append(1)
            TP_ids.append(y_test_id[idx])
        elif y == 1.0 and y != y_pred[idx]:
            FN.append(1)
            FN_ids.append(y_test_id[idx])
        elif y == 0.0 and y == y_pred[idx]:
            TN.append(1)
            TN_ids.append(y_test_id[idx])
        elif y == 0.0 and y != y_pred[idx]:
            FP.append(1)
            FP_ids.append(y_test_id[idx])

    # Current fold performance
    sensitivity = len(TP)/(len(TP)+len(FN))
    NPV = len(TN)/(len(TN)+len(FN))
    specificity = len(TN)/(len(TN)+len(FP))
    PPV = len(TP)/(len(TP)+len(FP))

    # Performance across folds
    mean_sensitivity.append(sensitivity)
    mean_specificity.append(specificity)
    mean_PPV.append(PPV)
    mean_NPV.append(NPV)

    # For plotting
    X_test_list.append(X_test)
    y_test_list.append(y_test)
    probas_ = clf.predict_proba(X_test)
    probas_list.append(probas_)

# Determine feature selection frequency across folds.
feats = [j for i in best_features for j in i]
feature_freq = {}
for idx, val in enumerate(feats):
    if val not in feature_freq.keys():
        feature_freq[val] = 1
    else:
        feature_freq[val] = feature_freq[val] + 1

# Performance visulazation
plt.figure(figsize=(10, 10), dpi=500)

# ROC for each fold
tprs = []
aucs = []
for idx, probas_ in enumerate(probas_list):
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y_test_list[idx], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    loop = idx + 1
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold %d (AUC = %0.2f)' % (loop, roc_auc))

# Mean ROC and confidence interval
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 Standard Deviation')

# Chance
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")

# Graph Labels
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

sensitivity = len(TP_ids)/(len(TP_ids)+len(FN_ids))
specificity = len(TN_ids)/(len(TN_ids)+len(FP_ids))
PPV = len(TP_ids)/(len(TP_ids)+len(FP_ids))
NPV = len(TN_ids)/(len(TN_ids)+len(FN_ids))

# Print performance
print('sensitivity:')
print(sensitivity)
print('specificity')
print(specificity)
print('PPV: ')
print(PPV)
print('NPV: ')
print(NPV)
print()
print('TPs:')
print(TP_ids)
print('TNs:')
print(TN_ids)
print('FPs:')
print(FP_ids)
print('FNs:')
print(FN_ids)
print()
print(best_parameters)
