import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from torch import nn
import torch
import copy
from time import time
from utils.CentroidLoss import CentroidLoss, cluster_accuracy
from sklearn import cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def introduce_label_noise(y, epsilon, n_classes):

    min_y = min(y)
    tmp_y = np.array([(i - min_y) for i in y])

    # Number of samples
    n_samples = tmp_y.shape[0]
    # Create the noise transition matrix

    T = np.full((n_classes, n_classes), epsilon / (n_classes - 1))
    np.fill_diagonal(T, 1 - epsilon)
    
    # Apply noise
    y_noisy = np.empty_like(tmp_y)
    for i in range(n_samples):
        current_label = tmp_y[i]
        y_noisy[i] = np.random.choice(n_classes, p=T[current_label])

    y_noisy = [(i + min_y) for i in y_noisy]

    return y_noisy

def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    ax.set_title(class_name)

def train_ae(model, n_epochs, criterion, train_dataset, val_dataset, logger):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  all_train_loss_ae = []
  all_val_loss_ae = []
  for epoch in range(1, n_epochs + 1):
    epoch_time = time()

    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss_ae = criterion(seq_pred, seq_true)

      loss_ae.backward()
      optimizer.step()

      train_losses.append(loss_ae.item())

    val_losses = []
    model = model.eval()

    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss_ae = criterion(seq_pred, seq_true)
        val_losses.append(loss_ae.item())

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    train_loss_ae = np.mean(train_losses)
    val_loss_ae = np.mean(val_losses)

    all_train_loss_ae.append(train_loss_ae)
    all_val_loss_ae.append(val_loss_ae)

    if val_loss_ae < best_loss:
      best_loss = val_loss_ae
      best_model_wts = copy.deepcopy(model.state_dict())

    epoch_time = time() - epoch_time
    logger.info(f'Epoch {epoch}: AE TL: {train_loss_ae:.3f}, AE VL:{val_loss_ae:.3f}, Epoch time: {epoch_time:.3f}s' )

  return model, best_model_wts, all_train_loss_ae, all_val_loss_ae

def train_class(embeddings_train, embeddings_val, model_class, n_epochs, criterion_class, train_target, train_dataset, val_target, val_dataset, n_classes, logger):
  optimizer = torch.optim.Adam(model_class.parameters(), lr=5e-3)

  best_model_wts = copy.deepcopy(model_class.state_dict())
  best_loss = 10000.0
  
  all_train_loss_class = []
  all_val_loss_class = []
  for epoch in range(1, n_epochs + 1):
    epoch_time = time()

    model_class = model_class.train()

    train_losses = []
    target_idx = 0
    train_targets = np.array(train_target)
    for embedding in embeddings_train:
      optimizer.zero_grad()

      y_pred = model_class(embedding)

      loss_class = criterion_class(y_pred, torch.tensor([train_targets[target_idx]/n_classes], dtype=torch.float32, device = device))
      target_idx += 1

      loss_class.backward()
      optimizer.step()

      train_losses.append(loss_class.item())

    val_losses = []

    model_class = model_class.eval()

    val_targets = np.array(val_target)
    target_idx = 0
    with torch.no_grad():
      for embedding in embeddings_val:

        y_pred = model_class(embedding)

        loss_class = criterion_class(y_pred, torch.tensor([val_targets[target_idx]/n_classes], dtype=torch.float32, device = device))
        target_idx += 1
        val_losses.append(loss_class.item())

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    train_loss_class = np.mean(train_losses)
    val_loss_class = np.mean(val_losses)

    all_train_loss_class.append(train_loss_class)
    all_val_loss_class.append(val_loss_class)

    if val_loss_class < best_loss:
      best_loss = val_loss_class
      best_model_wts = copy.deepcopy(model_class.state_dict())

    epoch_time = time() - epoch_time
    logger.info(f'Epoch {epoch}: CLS TL: {train_loss_class:.3f}, CLS VL:{val_loss_class:.3f}, Epoch time: {epoch_time:.3f}s' )

  return model_class, best_model_wts, all_train_loss_class, all_val_loss_class


def train_model(model, model_class, train_dataset, val_dataset, criterion, criterion_class, logger, embedding_size, n_classes, train_target, val_target, seed, n_epochs_ae, n_epoch_class):

  history = dict(train_ae=[], train_class=[], val_ae=[], val_class=[])

  train_time = time()

  # model, best_model_wts, train_loss_ae, val_loss_ae = train_ae(model, n_epochs_ae, criterion, train_dataset, val_dataset, logger)
  # model.load_state_dict(best_model_wts)
  # history['train_ae'].extend(train_loss_ae)
  # history['val_ae'].extend(val_loss_ae)
  model = torch.load('/home/sasisekhar/Desktop/SREA/modified_SREA/debug/Plane/08_04_2024__09_44_59/noise_0.3/model_ae.pth')

  model = model.eval()
  embeddings_train = []
  with torch.no_grad():
    for seq_true in train_dataset:
      seq_true = seq_true.to(device)
      embedding = model.encoder(seq_true).squeeze().to(device)
      embeddings_train.append(embedding)
  
  embeddings_val = []
  with torch.no_grad():
    for seq_true in val_dataset:
      seq_true = seq_true.to(device)
      embedding = model.encoder(seq_true).squeeze().to(device)
      embeddings_val.append(embedding)

  model_class, best_model_class_wts, train_loss_class, val_loss_class = train_class(embeddings_train, embeddings_val, model_class, n_epoch_class, criterion_class, train_target, train_dataset, val_target, val_dataset, n_classes, logger)
  model_class.load_state_dict(best_model_class_wts)
  history['train_class'].extend(train_loss_class)
  history['val_class'].extend(val_loss_class)


  train_time = time() - train_time
  logger.info(f'Model Train Time: {train_time}s')
  return model.eval(), history

def predict(model, dataset, criterion):
  predictions, losses = [], []
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

def plot_prediction(data, criterion, model, title, ax):
  predictions, pred_losses = predict(model, [data], criterion)

  ax.plot(data, label='true')
  ax.plot(predictions[0], label='reconstructed')
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
  ax.legend()
  return pred_losses
  
def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features, n_seq