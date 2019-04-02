import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import *

def ConfusionMatrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def Accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def SimpleMetrics(y_true, y_pred):
    print('Confusion Matrix:')
    print(ConfusionMatrix(y_true, y_pred))
    print('Accuracy: {}'.format(Accuracy(y_true, y_pred)))


def get_data_batch(train, batch_size, seed=0):
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(train)
    np.random.seed(shuffle_seed)
    train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
    x = train.loc[train_ix[start_i: stop_i]].values

    return np.reshape(x, (batch_size, -1))


def CheckAccuracy(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2):
    dtrain = np.vstack([x[:int(len(x) / 2)], g_z[:int(len(g_z) / 2)]])  # Use half of each real and generated set for training
    dlabels = np.hstack([np.zeros(int(len(x) / 2)), np.ones(int(len(g_z) / 2))])  # synthetic labels
    dtest = np.vstack([x[int(len(x) / 2):], g_z[int(len(g_z) / 2):]])  # Use the other half of each set for testing
    y_true = dlabels  # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes


    dtrain = xgb.DMatrix(dtrain, dlabels, feature_names=data_cols + label_cols)
    dtest = xgb.DMatrix(dtest, feature_names=data_cols + label_cols)

    xgb_params = {
        # 'tree_method': 'hist', # for faster evaluation
        'max_depth': 4,  # for faster evaluation
        'objective': 'binary:logistic',
        'random_state': 0,
        'eval_metric': 'auc',  # allows for balanced or unbalanced classes
    }
    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10)  # limit to ten rounds for faster evaluation

    y_pred = np.round(xgb_test.predict(dtest))

    # return '{:.2f}'.format(SimpleAccuracy(y_pred, y_true)) # assumes balanced real and generated datasets
    return Accuracy(y_pred, y_true)  # assumes balanced real and generated datasets


def PlotData(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix=''):
    real_samples = pd.DataFrame(x, columns=data_cols + label_cols)
    gen_samples = pd.DataFrame(g_z, columns=data_cols + label_cols)

    f, axarr = plt.subplots(1, 2, figsize=(6, 2))
    if with_class:
        axarr[0].scatter(real_samples[data_cols[0]], real_samples[data_cols[1]],
                         c=real_samples[label_cols[0]] / 2)  # , cmap='plasma'  )
        axarr[1].scatter(gen_samples[data_cols[0]], gen_samples[data_cols[1]],
                         c=gen_samples[label_cols[0]] / 2)  # , cmap='plasma'  )

        # For when there are multiple one-hot encoded label columns
        # for i in range(len(label_cols)):
        # temp = real_samples.loc[ real_samples[ label_cols[i] ] == 1 ]
        # axarr[0].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )
        # temp = gen_samples.loc[ gen_samples[ label_cols[i] ] == 1 ]
        # axarr[1].scatter( temp[data_cols[0]], temp[data_cols[1]], c='C'+str(i), label=i )

    else:
        axarr[0].scatter(real_samples[data_cols[0]], real_samples[data_cols[1]])  # , cmap='plasma'  )
        axarr[1].scatter(gen_samples[data_cols[0]], gen_samples[data_cols[1]])  # , cmap='plasma'  )
    axarr[0].set_title('real')
    axarr[1].set_title('generated')
    axarr[0].set_ylabel(data_cols[1])  # Only add y label to left plot
    for a in axarr: a.set_xlabel(data_cols[0])  # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(
        axarr[0].get_ylim())  # Use axes ranges from real data for generated data

    if save:
        plt.save(prefix + '.xgb_check.png')

    plt.show()

#### Functions to define the layers of the networks used in the 'define_models' functions below

class Gen(nn.Module):               # todo build generator without label
    def __init__(self):
        super(Gen, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(base_n_count, base_n_count*2),
            nn.ReLU(),
            nn.Linear(base_n_count*2, base_n_count*4),
            nn.ReLU(),
            nn.Linear(base_n_count*4, data_dim),
            nn.Sigmoid()
        )
        self.model.apply(xavier_init)


    def forward(self, x):
        return self.model(x)


#### Functions to define the keras network models

def define_models_GAN(rand_dim, data_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(rand_dim,))
    generated_image_tensor = generator_network(generator_input_tensor, data_dim, base_n_count)

    generated_or_real_image_tensor = layers.Input(shape=(data_dim,))

    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim, base_n_count)

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')

    return generator_model, discriminator_model, combined_model
