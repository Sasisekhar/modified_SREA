import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
import math
from datetime import datetime as dt
from pathlib import Path
import logging

from utils.LSTM_AE import RecurrentAutoencoder
from utils.ClassModel import NonLinClassifier
import utils.helpers as util
import utils.metrics as metrics


dataset     = 'PowerCons'
# class_names = ['Brouke Street Mall', 'Southern Cross Section', 'New Quay', 'Flinders', 'QV', 'Convention Centre', 'Chinatown', 'Webb Bridge', 'Tin', 'Southbank']
# class_names = ['C', 'B', 'F']
# class_names = ['Gomphonema augur', 'Fragilariforma bicapitata', 'Stauroneis smithii', 'Eunotia tenella']
# class_names = [s +  1 for s in range(24)]
class_names = ['Warm season', 'Cold Season']
# class_names = ['Desktop', 'Laptop']
# class_names = ['Acer Circinatum', 'Acer Glabrum', 'Acer Macrophyllum', 'Acer Negundo', 'Quercus Garryana', 'Quercus Kelloggii']
# class_names = ['Fridge/Freezer', 'Refrigerator', 'Upright Freezer']
# class_names = ['Avonlea', 'Clovis', 'Mix']
# class_names = ['Mirage', 'Eurofighter', 'F-14 wings closed', 'F-14 wings opened', 'Harrier', 'F-22', 'F-15']
# class_names = ['Ulmus carpinifolia', 'Acer', 'Salix aurita', 'Quercus', 'Alnus incana', 'Betula pubescens', 'Salix alba Sericea', 'Populus tremula', 'Ulmus glabra', 'Sorbus aucuparia', 'Salix sinerea',  'Populus', 'Tilia', 'Sorbus intermedia', 'Fagus silvatica']
n_epochs_ae = 500
n_epochs_class = 1000
n_classes = len(class_names)
embedding_dims = [2, 24, 48, 64, 128]

RESULT_PATH = 'debug/' + dataset + '/' + dt.now().strftime("%d_%m_%Y__%H_%M_%S")
# MODEL_AE_PATH  = RESULT_PATH + '/model_ae.pth'
# MODEL_CLS_PATH  = RESULT_PATH + '/model_cls.pth'

Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################MAIN##################################

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 710
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_df  = pd.read_csv("../UCRArchive_2018/" + dataset + "/" + dataset + "_TRAIN.tsv", sep='\t', header=None)
pd.concat([train_df, pd.read_csv("../UCRArchive_2018/" + dataset + "/" + dataset + "_TEST.tsv", sep='\t', header=None)])

train_df, test_df = train_test_split(
  train_df,
  test_size=0.3,
  random_state=RANDOM_SEED
)

new_columns = list(train_df.columns)
new_columns[0] = 'target'
train_df.columns = new_columns

new_columns = list(test_df.columns)
new_columns[0] = 'target'
test_df.columns = new_columns

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create logger
logger = logging.getLogger(__name__)

# Create file handler and set level to debug
file_handler = logging.FileHandler(RESULT_PATH + "/out.log")
file_handler.setLevel(logging.INFO)

# Add handlers to logger
# logger.addHandler(console_handler)
logger.addHandler(file_handler)
#################################Data Visualization################################################

# Time Series Plots:
classes = train_df.target.unique()
num_rows = len(classes) // 3 + 1
num_cols = 6  # 3 pairs of training and testing plots per row

fig, axs = plt.subplots(
    nrows=num_rows,
    ncols=num_cols,
    figsize=(15, num_rows * 3),  # Adjust figure size as needed
    sharey=True,
)

for i, cls in enumerate(classes):
    train_ax = axs.flat[i*2]  # Even indices for training
    test_ax = axs.flat[i*2 + 1]  # Odd indices for testing

    # Training data plot
    train_data = train_df[train_df.target == cls] \
        .drop(labels='target', axis=1) \
        .mean(axis=0) \
        .to_numpy()
    util.plot_time_series_class(train_data, f"Train: {class_names[i]}", train_ax)

    # Testing data plot
    test_data = test_df[test_df.target == cls] \
        .drop(labels='target', axis=1) \
        .mean(axis=0) \
        .to_numpy()
    util.plot_time_series_class(test_data, f"Test: {class_names[i]}", test_ax)

# Remove any unused axes
for j in range(len(classes)*2, num_rows*num_cols):
    fig.delaxes(axs.flat[j])

fig.tight_layout()
plt.savefig(RESULT_PATH + '/1_Time_Series_plot.png')

# Frequency Plots
train_label_counts = train_df.target.value_counts().sort_index()
test_label_counts = test_df.target.value_counts().sort_index()

# Setup the figure and subplots
plt.figure(figsize=(20, 6))  # Increase figure size to accommodate both plots side by side

# Plot for training dataset
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
train_label_counts.plot(kind='bar')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Frequency of Class Labels in the Training Dataset')
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)

# Plot for testing dataset
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
test_label_counts.plot(kind='bar')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Frequency of Class Labels in the Testing Dataset')
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=90)

plt.tight_layout()  # Adjust layout to not overlap
plt.savefig(RESULT_PATH + '/2_Frequency_plot.png')

#################################Train Model#########################################

train_df, val_df = train_test_split(
  train_df,
  test_size=0.3,
  random_state=RANDOM_SEED
)

epsilon = 0.0

# train_targets = train_df.target
# val_targets = val_df.target
# test_targets = test_df.target


train_targets_actual = train_df.target
train_targets = util.introduce_label_noise(train_targets_actual, epsilon)

val_targets_actual = val_df.target
val_targets = util.introduce_label_noise(val_targets_actual, epsilon)

test_targets_actual = test_df.target
test_targets = util.introduce_label_noise(test_targets_actual, epsilon)

targets_actual = list(train_targets_actual) + list(val_targets_actual) + list(test_targets_actual)
targets_noisy = train_targets + val_targets + test_targets

cm = confusion_matrix(targets_actual, targets_noisy)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.tight_layout()
plt.savefig(RESULT_PATH + '/2.5_Noise_Confusion_Matrix.png')

train_df = train_df.drop(labels='target', axis=1)
val_df = val_df.drop(labels='target', axis=1)
test_df = test_df.drop(labels='target', axis=1)

train_dataset, seq_len, n_features, _ = util.create_dataset(train_df)
val_dataset, _, _, _ = util.create_dataset(val_df)
test_dataset, _, _, _ = util.create_dataset(test_df)

criterion = nn.HuberLoss(reduction='mean').to(device)
criterion_class = nn.MSELoss(reduction='mean').to(device)

class_results = []
ae_results = []

for embedding_dim in embedding_dims:
  EMBEDDING_PATH = RESULT_PATH + '/embedding_dim_' + str(embedding_dim)
  Path(EMBEDDING_PATH).mkdir(parents=True, exist_ok=True)

  model_class = NonLinClassifier(embedding_dim).to(device)
  model = RecurrentAutoencoder(seq_len, n_features, embedding_dim).to(device)

  logger.info(f'Dataset: {dataset}, n_epoch_ae: {n_epochs_ae}, n_epoch_class: {n_epochs_class}, embedding_dim: {embedding_dim}, n_classes: {n_classes}')

  model, history = util.train_model(
    model,
    model_class,
    train_dataset, 
    val_dataset,
    criterion,
    criterion_class,
    logger,
    embedding_dim,
    n_classes,
    train_targets,
    val_targets,
    RANDOM_SEED,
    n_epochs_ae,
    n_epochs_class
  )

  ax = plt.figure().gca()

  ax.plot(history['train_ae'])
  ax.plot(history['train_class'])
  ax.plot(history['val_ae'])
  ax.plot(history['val_class'])

  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train_ae', 'train_class', 'val_ae', 'val_class'])
  plt.title('Loss over training epochs')
  plt.savefig(EMBEDDING_PATH + '/3_Loss.png')

  torch.save(model, EMBEDDING_PATH + '/model_ae.pth')
  torch.save(model_class, EMBEDDING_PATH + '/model_class.pth')

  # model = torch.load('/home/sasisekhar/Desktop/SREA/modified_SREA/results/PowerCons/30_03_2024__17_19_17/model_ae.pth')
  # model_class = torch.load('/home/sasisekhar/Desktop/SREA/modified_SREA/results/PowerCons/30_03_2024__17_19_17/model_cls.pth')
  model_class = model_class.eval()
  model = model.eval()

  target_idx = 0
  test_metric_vals = {"AE_actual": [], "AE_predicted": [], "class_predicted": []}
  with torch.no_grad():
    for seq_true in test_dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      test_metric_vals['AE_actual'].extend(seq_true.cpu().numpy())
      test_metric_vals['AE_predicted'].extend(seq_pred.cpu().numpy())
      embedding = model.encoder(seq_true).squeeze().to(device)
      y_pred = model_class(embedding)

      test_metric_vals['class_predicted'].append(y_pred.item())

  test_metric_vals['class_predicted'] = [int(np.round(i*n_classes)) for i in test_metric_vals['class_predicted']] #Scaling the labels from [0, 1] to [0, +inf)

  ae_metrics, class_metrics = metrics.evaluate( np.array(test_metric_vals['AE_actual']), 
                                                np.array(test_metric_vals['AE_predicted']),
                                                np.array(test_targets),
                                                np.array(test_targets_actual),
                                                np.array(test_metric_vals['class_predicted']),
                                                logger,
                                                class_names,
                                                metrics=['mae', 'mse', 'rmse', 'std_ae', 'smape', 'rae', 'mbrae', 'r2', 'dtw'])
  
  class_results.append(class_metrics)
  ae_results.append(ae_metrics)

  cm = confusion_matrix(test_targets, test_metric_vals['class_predicted'])
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot()
  plt.tight_layout()
  plt.savefig(EMBEDDING_PATH + '/4_Confusion_Matrix_NOISE.png')

  cm = confusion_matrix(test_targets_actual, test_metric_vals['class_predicted'])
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot()
  plt.tight_layout()
  plt.savefig(EMBEDDING_PATH + '/4.5_Confusion_Matrix_TRUE.png')


  fig, axs = plt.subplots(
    nrows= min(4, math.ceil(len(test_dataset)/4)),
    ncols=4,
    sharey=True,
    sharex=True,
    figsize=(22, 8)
  )

  for i, data in enumerate(test_dataset[: min(4, math.ceil(len(test_dataset)/4)) * 4]):
    util.plot_prediction(data, criterion, model, title=dataset, ax=axs[i//4, i%4])

  fig.tight_layout()
  plt.savefig(EMBEDDING_PATH + '/5_Reconstructed.png')

# Metric plot for different embedding dims

# Preparing data for plotting
labels = class_results[0].keys()
non_label_metrics = ['accuracy', 'macro avg', 'weighted avg']
tmp = [x for x in labels if x not in non_label_metrics]
labels = tmp
num_labels = len(labels)

fig, axes = plt.subplots(num_labels, 3, figsize=(18, 6*num_labels))

plot_metrics = ['precision', 'recall', 'f1-score']
embedding_dims_x = embedding_dims

for row, label in enumerate(labels):
    for col, metric in enumerate(plot_metrics):
        y_values = [result[label][metric] for result in class_results]
        axes[row, col].plot(embedding_dims, y_values, label=f'{label} {metric}', marker='o')
        axes[row, col].set_title(f'{label} {metric.capitalize()} vs Embedding Dimensions')
        axes[row, col].set_xlabel('Embedding Dimensions')
        axes[row, col].set_ylabel(metric.capitalize())
        axes[row, col].legend()

plt.tight_layout()
plt.savefig(RESULT_PATH + '/6_Class_Metrics.png')

# Creating a 1x4 plot for ae_results
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# ae_metrics and their corresponding subplot titles
ae_metrics = ['mae', 'mse', 'r2', 'dtw']
plot_titles = ['MAE', 'MSE', 'R^2', 'Dynamic Time Warping']

for i, (metric, title) in enumerate(zip(ae_metrics, plot_titles)):
    y_values = [result[metric] for result in ae_results]
    axes[i].plot(embedding_dims, y_values, label=title, marker='o')
    axes[i].set_title(f'{title} vs Embedding Dimensions')
    axes[i].set_xlabel('Embedding Dimensions')
    axes[i].set_ylabel(metric.upper())
    axes[i].legend()

plt.tight_layout()
plt.savefig(RESULT_PATH + '/7_AE_Metrics.png')
