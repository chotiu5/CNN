from pyfasta import Fasta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Input, Dense,Lambda,Conv1D,Flatten,AveragePooling1D
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import manifold, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import matplotlib as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc



f = Fasta("SARS-CoV-2.fa")

# Data preprocessing
sorted(f.keys())

key=sorted(f.keys())
aa=[]
Xp=[]
for name in sorted(f.keys()):
    k=list(f[name][:])
    Xp.append(k)

Xp=pd.DataFrame(Xp)
Xp

one=np.ones(2000)
Y=np.concatenate([0*one,one,2*one,3*one])
len(Y)

Xp=Xp.replace(['N','A','G','C','T','K','W','Y','R','M','S'],[0,1,2,3,4,5,6,7,8,9,10])
Xp=Xp.iloc[:,0:28919]
Xp

# split the whole dataset
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# test is now 10% of the entire data set
# Train would be used for 10-fold or final evaluation
x_Train, x_test, y_Train, y_test = train_test_split(Xp, Y, test_size=test_ratio, stratify=Y, random_state=12345)

# train is now 80% of the initial data set
# validation is now 10% of the initial data set
x_train, x_val, y_train, y_val = train_test_split(x_Train, y_Train, test_size=validation_ratio/(validation_ratio + train_ratio), stratify=y_Train, random_state=12345)

x_train.shape

x_train

y4_train=to_categorical(y_train)
y4_Train=to_categorical(y_Train)
y4_val=to_categorical(y_val)
y4_test=to_categorical(y_test)
x4_train=np.array(x_train).astype(int).reshape((6400,28919,1))
x4_Train=np.array(x_Train).astype(int).reshape((7200,28919,1))
x4_val=np.array(x_val).astype(int).reshape((800,28919,1))
x4_test=np.array(x_test).astype(int).reshape((800,28919,1))

def cnn_epoch(x4_train, y4_train, x4_val, y4_val):
    input_img = Input(shape=(28919,1))
    x = Conv1D(16, 5,activation='relu',padding='same')(input_img)
    x= AveragePooling1D(pool_size=2)(x)
    x = Conv1D(16, 5,activation='relu',padding='same')(x)
    x= AveragePooling1D(pool_size=2)(x)
    x = Conv1D(16, 5,activation='relu',padding='same')(x)
    x= AveragePooling1D(pool_size=2)(x)
    x=Flatten()(x)
    encoded=Dense(1000, activation='relu',use_bias=False)(x)
    decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(encoded)
    autoencoder1 = Model(inputs=input_img, outputs=decoded)
    encoder1 = Model(inputs=input_img, outputs=encoded)
    autoencoder1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    autoencoder1.fit(x4_train, y4_train,
                     epochs=70,
                     batch_size=64,
                     shuffle=True,
                     validation_data=(x4_val, y4_val))

cnn_epoch(x4_train, y4_train, x4_val, y4_val)

#stratified 10-fold validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
kf.get_n_splits(x4_Train, y_Train)

def cnn_10fold(x4_Train, y_Train):
    for k, (train, test) in enumerate(kf.split(x4_Train, y_Train)):
        input_img = Input(shape=(28919,1))
        x = Conv1D(16, 5,activation='relu',padding='same')(input_img)
        x= AveragePooling1D(pool_size=2)(x)
        x = Conv1D(16, 5,activation='relu',padding='same')(x)
        x= AveragePooling1D(pool_size=2)(x)
        x = Conv1D(16, 5,activation='relu',padding='same')(x)
        x= AveragePooling1D(pool_size=2)(x)
        x=Flatten()(x)
        encoded=Dense(1000, activation='relu',use_bias=False)(x)
        decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(encoded)
        autoencoder1 = Model(inputs=input_img, outputs=decoded)
        encoder1 = Model(inputs=input_img, outputs=encoded)
        autoencoder1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        y4_Train=to_categorical(y_Train)
        autoencoder1.fit(x4_Train[train], y4_Train[train],
                         epochs=23,
                         batch_size=64,
                         shuffle=True,
                         validation_data=(x4_Train[test], y4_Train[test]))
        # make predictions
        yhat = autoencoder1.predict(x4_Train[test])
        y_pred = np.argmax(yhat, axis=1)
        y_true = y_Train[test]
        print(metrics.classification_report(y_true, y_pred, digits=4))

# External validation and visualization of CNN classifier performance

input_img = Input(shape=(28919,1))
x = Conv1D(16, 5,activation='relu',padding='same')(input_img)
x= AveragePooling1D(pool_size=2)(x)
x = Conv1D(16, 5,activation='relu',padding='same')(x)
x= AveragePooling1D(pool_size=2)(x)
x = Conv1D(16, 5,activation='relu',padding='same')(x)
x= AveragePooling1D(pool_size=2)(x)
x=Flatten()(x)
encoded=Dense(1000, activation='relu',use_bias=False)(x)
decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(encoded)
autoencoder1 = Model(inputs=input_img, outputs=decoded)
encoder1 = Model(inputs=input_img, outputs=encoded)
autoencoder1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
autoencoder1.fit(x4_Train, y4_Train,
                 epochs=23,
                 batch_size=64,
                 shuffle=True,
                 validation_data=(x4_test, y4_test))
y4_score = autoencoder1.predict(x4_test)
y_pred = np.argmax(y4_score, axis=1)
y_true = y_test
print(metrics.classification_report(y_true, y_pred, digits=4))

# Confusion Matrix

cm = confusion_matrix(y4_true, y_pred)
plt.rcParams['figure.figsize'] = [8, 6]
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g',ax=ax, cmap=sns.light_palette("#0173b2",as_cmap=True), cbar_kws={'ticks': [0,50,100,150,200]})
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(["B.1.1.7", "B.1.427", "B.1.526","P.1"]); ax.yaxis.set_ticklabels(["B.1.1.7", "B.1.427", "B.1.526","P.1"])
plt.savefig('Confusion Matrix.jpg', format='jpg', dpi=1200)

# ROC curves

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='#878787',linestyle=':', linewidth=3)

plt.plot(fpr[0], tpr[0], color='#0173b2', lw=lw,
         label='ROC curve of B.1.1.7          (area = {1:0.2f})'
         ''.format(0, roc_auc[0]))

plt.plot(fpr[1], tpr[1], color='#de8f05', lw=lw,
         label='ROC curve of B.1.427         (area = {1:0.2f})'
         ''.format(1, roc_auc[1]))

plt.plot(fpr[2], tpr[2], color='#029e73', lw=lw,
         label='ROC curve of B.1.526         (area = {1:0.2f})'
         ''.format(0, roc_auc[2]))

plt.plot(fpr[3], tpr[3], color='#d55e00', lw=lw,
         label='ROC curve of P.1                 (area = {1:0.2f})'
         ''.format(0, roc_auc[3]))

plt.rcParams['figure.figsize'] = [8, 6]
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.025, 1.0])
plt.ylim([-0.025, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.legend(loc="lower right")
plt.grid(False)
plt.tick_params(labelsize=15)
plt.savefig('ROC.jpg', format='jpg', dpi=1200)

# t-SNE two-dimensional visualization of SARS-CoV-2 genome dataset.

def data_visualization(data,label,filename,title):
    # Transform data
    tsne = manifold.TSNE(n_components=2, perplexity = 38)
    X_tsne = tsne.fit_transform(data)
    # Plotting
    plt.rcParams['figure.figsize'] = [8, 6]
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    rawplot = sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=label, palette="colorblind", legend = True, s = 50,edgecolor="none", alpha=.55)
    legend_labels, _= rawplot.get_legend_handles_labels()
    rawplot.legend(legend_labels, ['B.1.1.7', 'B.1.427','B.1.526','P.1'], loc = "upper right")
    plt.xlabel('Dimension 1', fontsize=15)
    plt.ylabel('Dimension 2', fontsize=15)
    rawplot.set_title(title)
    plt.savefig(filename, format='jpg', dpi=1200)

data_visualization(x_Train,y_Train,'rawtrain.jpg','')

data_visualization(x_test,y_test,'rawtest.jpg','')

trained_train = encoder1.predict(x_Train)
data_visualization(trained_train,y_Train,'trained_Train.jpg','')

trained_test = encoder1.predict(x_test)
data_visualization(trained_test,y_test,'trained_test.jpg','')

#  other machine learning methods

#  Logistic Regression
def lr_cross_val(x_Train, y_Train, x_test, y_test):
    x2_Train=np.array(x_Train).astype(int)
    x2_test=np.array(x_test).astype(int)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    kf.get_n_splits(x2_Train, y_Train)
    for k, (train, test) in enumerate(kf.split(x2_Train, y_Train)):
        clf = linear_model.LogisticRegression(max_iter=5000)
        lr_train= clf.fit(x2_Train[train], y_Train[train])
        y_pred= lr_train.predict(x2_Train[test])
        y_true = y_Train[test]
        print("Classification_report for 10 fold:")
        print(metrics.classification_report(y_true, y_pred, digits=4))
    Y_pred = linear_model.LogisticRegression(max_iter=5000).fit(x2_Train, y_Train).predict(x2_test)
    Y_true = y_test
    print("Classification_report for test:")
    print(metrics.classification_report(Y_true, Y_pred, digits=4))

lr_cross_val(x_Train, y_Train, x_test, y_test)

# K-Nearest Neighbor

def knn_cross_val(x_Train, y_Train, x_test, y_test):
    x2_Train=np.array(x_Train).astype(int)
    x2_test=np.array(x_test).astype(int)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    kf.get_n_splits(x2_Train, y_Train)
    for k, (train, test) in enumerate(kf.split(x2_Train, y_Train)):
        clf = neighbors.KNeighborsClassifier()
        knn_train= clf.fit(x2_Train[train], y_Train[train])
        y_pred= knn_train.predict(x2_Train[test])
        y_true = y_Train[test]
        print("Classification_report for 10 fold:")
        print(metrics.classification_report(y_true, y_pred,digits=4))
    Y_pred = neighbors.KNeighborsClassifier().fit(x2_Train, y_Train).predict(x2_test)
    Y_true = y_test
    print("Classification_report for test:")
    print(metrics.classification_report(Y_true, Y_pred, digits=4))

knn_cross_val(x_Train, y_Train, x_test, y_test)

# Support Vector Machine
def svm_cross_val(x_Train, y_Train, x_test, y_test):
    x2_Train=np.array(x_Train).astype(int)
    x2_test=np.array(x_test).astype(int)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    kf.get_n_splits(x2_Train, y_Train)
    for k, (train, test) in enumerate(kf.split(x2_Train, y_Train)):
        clf = svm.SVC()
        svm_train= clf.fit(x2_Train[train], y_Train[train])
        y_pred= svm_train.predict(x2_Train[test])
        y_true = y_Train[test]
        print("Classification_report for 10 fold:")
        print(metrics.classification_report(y_true, y_pred,digits=4))
    Y_pred = svm.SVC().fit(x2_Train, y_Train).predict(x2_test)
    Y_true = y_test
    print("Classification_report for test:")
    print(metrics.classification_report(Y_true, Y_pred, digits=4))

svm_cross_val(x_Train, y_Train, x_test, y_test)

# Random Forest

def rf_cross_val(x_Train, y_Train, x_test, y_test):
    x2_Train=np.array(x_Train).astype(int)
    x2_test=np.array(x_test).astype(int)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    kf.get_n_splits(x2_Train, y_Train)
    for k, (train, test) in enumerate(kf.split(x2_Train, y_Train)):
        clf = ensemble.RandomForestClassifier()
        rf_train= clf.fit(x2_Train[train], y_Train[train])
        y_pred= rf_train.predict(x2_Train[test])
        y_true = y_Train[test]
        print("Classification_report for 10 fold:")
        print(metrics.classification_report(y_true, y_pred,digits=4))
    Y_pred = ensemble.RandomForestClassifier().fit(x2_Train, y_Train).predict(x2_test)
    Y_true = y_test
    print("Classification_report for test:")
    print(metrics.classification_report(Y_true, Y_pred, digits=4))

rf_cross_val(x_Train, y_Train, x_test, y_test)

# Multilayer perceptron

def MLP_10fold(x4_Train, y_Train):
    for k, (train, test) in enumerate(kf.split(x4_Train, y_Train)):
        input_img = Input(shape=(28919,))
        x = Dense(1000, activation='relu',use_bias=False)(input_img)
        decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(x)
        autoencoder = Model(inputs=input_img, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        y4_Train=to_categorical(y_Train)
        autoencoder.fit(x4_Train[train], y4_Train[train],
                        epochs=23,
                        batch_size=64,
                        shuffle=True,
                        validation_data=(x4_Train[test], y4_Train[test]))
        yhat = autoencoder.predict(x4_Train[test])
        y_pred = np.argmax(yhat, axis=1)
        y_true = y_Train[test]
        print(metrics.classification_report(y_true, y_pred, digits=4))

MLP_10fold(x4_Train, y_Train)

def MLP_test(x4_Train, y4_Train, x4_test, y4_test):
    input_img = Input(shape=(28919,))
    x = Dense(1000, activation='relu',use_bias=False)(input_img)
    decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(x)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    autoencoder.fit(x4_Train, y4_Train,
                epochs=23,
                batch_size=64,
                shuffle=True,
                validation_data=(x4_test, y4_test))
    # make predictions for test data
    yhat = autoencoder.predict(x4_test)
    y_pred = np.argmax(yhat, axis=1)
    y_true = np.argmax(y4_test, axis=1)
    print(metrics.classification_report(y_true, y_pred,digits=4))

MLP_test(x4_Train, y4_Train, x4_test, y4_test)

def MLP_epoch(x4_train, y4_train, x4_val, y4_val):
    input_img = Input(shape=(28919,))
    x = Dense(1000, activation='relu',use_bias=False)(input_img)
    decoded= Dense(4, activation='softmax',use_bias=False,name="Dense_4")(x)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    autoencoder.fit(x4_train, y4_train,
                    epochs=100,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x4_val, y4_val))

MLP_epoch(x4_train, y4_train, x4_val, y4_val)
