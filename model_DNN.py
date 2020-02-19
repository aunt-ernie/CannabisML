import tempfile
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
# Feature Selection
from sklearn.feature_selection import f_classif, SelectPercentile
# Neural Network
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from clr_callback import CyclicLR
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import clear_session
from keras.optimizers import SGD
from AdamW import AdamW

# Parallel processing
from joblib import Memory
import warnings
warnings.filterwarnings("ignore")

# Read in covariate adjusted features (performed in R)
base_dir = '/mnt/Filbey/Ryan/currentProjects/SVM/Final/Combined/features'

# Diffusion data, thresholded at 0.3
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


# Custom class to allow shape of transformed x to be known to classifier
class ANOVASelection(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=10):
        self.percentile = percentile
        self.m = None
        self.X_new = None
        self.scores_ = None

    def fit(self, X, y):
        self.m = SelectPercentile(f_classif, self.percentile)
        self.m.fit(X, y)
        self.scores_ = self.m.scores_
        return self

    def transform(self, X):
        global X_new
        self.X_new = self.m.transform(X)
        X_new = self.X_new
        return self.X_new


# Define neural net architecture
def create_model(init='normal', activation_1='relu', activation_2='relu', optimizer='SGD', decay=0.1):
    clear_session()
    # Determine nodes in hidden layers (Huang et al., 2003)
    m = 1  # number of ouput neurons
    N = np.shape(data)[0]  # number of samples
    hn_1 = int(np.sum(np.sqrt((m+2)*N)+2*np.sqrt(N/(m+2))))
    hn_2 = int(m*np.sqrt(N/(m+2)))

    # Create layers
    model = Sequential()
    if optimizer == 'SGD':
        model.add(Dense(hn_1, input_dim=np.shape(X_new)[1], kernel_initializer=init, kernel_regularizer=regularizers.l2(decay), activation=activation_1))
        #model.add(Dropout(0.5))
        model.add(Dense(hn_2, kernel_initializer=init, kernel_regularizer=regularizers.l2(decay), activation=activation_2))
        #model.add(Dropout(0.5))
    elif optimizer == 'AdamW':
        model.add(Dense(hn_1, input_dim=np.shape(X_new)[1], kernel_initializer=init, activation=activation_1))
        #model.add(Dropout(0.5))
        model.add(Dense(hn_2, kernel_initializer=init, activation=activation_2))
        #model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    if optimizer == 'SGD':
        model.compile(loss='binary_crossentropy', optimizer=SGD(nesterov=True), metrics=["accuracy"])
    elif optimizer == 'AdamW':
        model.compile(loss='binary_crossentropy', optimizer=AdamW(weight_decay=decay), metrics=["accuracy"])
    return model


# Cyclic learning rate range test
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
for train_indices, test_indices in kfold.split(data, labels):
    X_train = [data[idx] for idx in train_indices]
    y_train = [labels[idx] for idx in train_indices]
    X_test = [data[idx] for idx in test_indices]
    y_test = [labels[idx] for idx in test_indices]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    break

bs = 128
ss = 4*np.ceil(X_train.shape[0]/bs)
clr = None
clr = CyclicLR(base_lr=0.001, max_lr=1, step_size=ss)
estimator = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
estimator.fit(X_train, y_train, batch_size=bs, callbacks=[clr])

h = clr.history
lr = h['lr']
acc = h['acc']
lr = h['lr'][:int(ss)+1]
acc = h['acc'][:int(ss)+1]
loss = h['loss'][:int(ss)+1]
plt_loss = []
plt_lr = []
delta_l = []
best_l = 1
for idx, l in enumerate(loss):
    if idx == 0 or l < best_l:
        best_l = l
    elif l > 2 * best_l:
        break
    delta_l.append(l - loss[idx-1])
    plt_loss.append(l)
    plt_lr.append(lr[idx])

for l in delta_l:
    if l < 0:
        first_dec = delta_l.index(l)
        break

lr_0 = lr[first_dec]
loss_0 = loss[first_dec]
lr_1 = lr[delta_l.index(min(delta_l))]
loss_1 = loss[delta_l.index(min(delta_l))]

# Plot accuracy as a function of learning rate.
plt.figure()
plt.figure(dpi=500)
plt.plot(lr, acc)
plt.axvline(x=lr_0, color="r", markersize=5)
plt.axvline(x=lr_1, color="r", markersize=5)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Learning Rate Range Test')

# Plot loss as a function of log(learning rate).
plt.figure()
plt.figure(dpi=500)
plt.xscale("log")
plt.plot(plt_lr, plt_loss)
plt.plot(lr_0, loss_0, marker='o', color="r", markersize=10)
plt.plot(lr_1, loss_1, marker='o', color="r", markersize=10)
plt.xlabel('log(Learning Rate)')
plt.ylabel('Loss')
plt.title('Learning Rate Range Test')


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

# Define Grid
tuned_parameters = dict(
                            ANOVA__percentile = [20, 40, 60, 80],
                            NN__optimizer = ['AdamW', 'SGD'],
                            NN__init = ['glorot_normal', 'glorot_uniform'],
                            NN__activation_1 = ['relu', 'sigmoid', 'tanh'],
                            NN__activation_2 = ['relu', 'sigmoid', 'tanh'],
                            NN__batch_size = [32, 64, 128, 256],
                            NN__decay = [10.0**i for i in range(-10,-1) if i%2 == 1]
                        )

# Nested CV
outer_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
inner_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
for train_indices, test_indices in outer_kfold.split(data, labels):
    # Ensure models from last iteration have been cleared.
    clear_session()

    # Split data
    X_train = [data[idx] for idx in train_indices]
    y_train = [labels[idx] for idx in train_indices]
    X_test = [data[idx] for idx in test_indices]
    y_test = [labels[idx] for idx in test_indices]
    y_test_id = [ids[idx] for idx in test_indices]

    # Apply mean and variance center based on training fold
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # New results lists for each fold
    TP = []
    TN = []
    FP = []
    FN = []

    # Memory handling
    cachedir = tempfile.mkdtemp()
    mem = Memory(location=cachedir, verbose=0)
    f_classif = mem.cache(f_classif)

    # Cyclic Learning Rate
    clr = CyclicLR(mode='triangular', base_lr=0.175, max_lr=0.9175, step_size=12)
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    # Build and train model
    ANOVA = ANOVASelection(percentile=5)
    NN = KerasClassifier(build_fn=create_model, epochs=1000, verbose=0)
    clf = Pipeline([('ANOVA', ANOVA), ('NN', NN)])
    clf = GridSearchCV(clf, tuned_parameters, scoring='balanced_accuracy', n_jobs=-1, cv=inner_kfold)
    clf.fit(X_train, y_train, NN__callbacks=[clr, es])
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

    # Free memory
    del clf
    clear_session()

# Determine best features that were found in > 1 fold.
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
