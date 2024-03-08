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

from utils.LSTM_AE import RecurrentAutoencoder
from utils.ClassModel import NonLinClassifier
import utils.helpers as util
from pathlib import Path
import logging

dataset     = 'OSULeaf'
# class_names = ['Brouke Street Mall', 'Southern Cross Section', 'New Quay', 'Flinders', 'QV', 'Convention Centre', 'Chinatown', 'Webb Bridge', 'Tin', 'Southbank']
# class_names = ['C', 'B', 'F']
# class_names = ['Gomphonema augur', 'Fragilariforma bicapitata', 'Stauroneis smithii', 'Eunotia tenella']
# class_names = [s + 1 for s in range(5)]
# class_names = ['Warm season', 'Cold Season']
class_names = ['Acer Circinatum', 'Acer Glabrum', 'Acer Macrophyllum', 'Acer Negundo', 'Quercus Garryana', 'Quercus Kelloggii']

n_epochs = 700
n_classes = len(class_names)
embedding_dim = 3

RESULT_PATH = 'results/' + dataset + '/' + dt.now().strftime("%d_%m_%Y__%H_%M_%S")
MODEL_AE_PATH  = RESULT_PATH + '/model_ae.pth'
MODEL_CLS_PATH  = RESULT_PATH + '/model_cls.pth'

Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################MAIN##################################

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_df  = pd.read_csv("../UCRArchive_2018/" + dataset + "/" + dataset + "_TRAIN.tsv", sep='\t', header=None)
test_df   = pd.read_csv("../UCRArchive_2018/" + dataset + "/" + dataset + "_TEST.tsv", sep='\t', header=None)

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
  test_size=0.15,
  random_state=RANDOM_SEED
)

train_targets = train_df.target
val_targets = val_df.target
test_targets = test_df.target

train_df = train_df.drop(labels='target', axis=1)
val_df = val_df.drop(labels='target', axis=1)
test_df = test_df.drop(labels='target', axis=1)

logger.info('Train Set:')
logger.info(train_df.head())
logger.info('Validate Set:')
logger.info(val_df.head())
logger.info('Test Set:')
logger.info(test_df.head())

train_dataset, seq_len, n_features, _ = util.create_dataset(train_df)
val_dataset, _, _, _ = util.create_dataset(val_df)
test_dataset, _, _, _ = util.create_dataset(test_df)

model_class = NonLinClassifier(embedding_dim).to(device)
model = RecurrentAutoencoder(seq_len, n_features, embedding_dim).to(device)

# criterion = nn.L1Loss(reduction='sum').to(device)
criterion = nn.HuberLoss(reduction='mean').to(device)
criterion_class = nn.MSELoss(reduction='mean').to(device)
# criterion = nn.MSELoss(reduction='mean').to(device)

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
  n_epochs
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
plt.savefig(RESULT_PATH + '/3_Loss.png')

torch.save(model, MODEL_AE_PATH)
torch.save(model_class, MODEL_CLS_PATH)

# model = torch.load('/home/sasisekhar/Desktop/SREA/Own_trial/results/PowerCons/07_03_2024__21_54_50/model_ae.pth')
# model_class = torch.load('/home/sasisekhar/Desktop/SREA/Own_trial/results/PowerCons/07_03_2024__21_54_50/model_cls.pth')
model_class = model_class.eval()
model = model.eval()

target_idx = 0
y_preds = []
with torch.no_grad():
  for seq_true in test_dataset:
    seq_true = seq_true.to(device)
    seq_pred = model(seq_true)
    embedding = model.encoder(seq_true).squeeze().to(device)
    y_pred = model_class(embedding)

    y_preds.append(y_pred.item())

y_preds = [int(np.round(i*n_classes)) for i in y_preds]

cm = confusion_matrix(test_targets, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.savefig(RESULT_PATH + '/4_Confusion_Matrix.png')


fig, axs = plt.subplots(
  nrows= min(4, math.ceil(len(test_dataset)/6)),
  ncols=6,
  sharey=True,
  sharex=True,
   figsize=(22, 8)
)

for i, data in enumerate(test_dataset[: min(4, math.ceil(len(test_dataset)/6)) * 6]):
  util.plot_prediction(data, criterion, model, title=dataset, ax=axs[i//6, i%6])

fig.tight_layout()
plt.savefig(RESULT_PATH + '/5_Reconstructed.png')