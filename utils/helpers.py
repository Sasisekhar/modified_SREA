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
  
def train_model(model, model_class, train_dataset, val_dataset, criterion, criterion_class, logger, embedding_size, n_classes, train_target, val_target, seed, n_epochs):
  # optimizer_ae = torch.optim.Adam(model.parameters(), lr=1e-4)
  # optimizer_class = torch.optim.Adam(model_class.parameters(), lr=1e-4)

  history = dict(train_ae=[], train_class=[], val_ae=[], val_class=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  # loss_centroids = CentroidLoss(embedding_size, n_classes, reduction='none').to(device)
  # kmeans = cluster.KMeans(n_clusters=n_classes, random_state=seed)

  optimizer = torch.optim.Adam(
        list(model.parameters()) + list(model_class.parameters()),
        lr=1e-3)
  
  train_time = time()
  for epoch in range(1, n_epochs + 1):
    epoch_time = time()

    # if epoch == 1:
    #     # Init cluster centers with KMeans
    #     embedding = []
    #     with torch.no_grad():
    #         model.eval()
    #         loss_centroids.eval()
    #         for seq_true in train_dataset:
    #             seq_true = seq_true.to(device)
    #             output = model.encoder(seq_true)
    #             embedding.append(output.squeeze().cpu().numpy())
    #     embedding = np.concatenate(embedding, axis=0)
    #     embedding = embedding.reshape(train_target.size, embedding_size)
    #     # embedding = np.stack((embedding, train_target), axis=1)
    #     predicted = kmeans.fit_predict(embedding)
    #     train_targets = np.array(train_target)
    #     train_targets = np.array([i-1 for i in train_targets])
    #     reassignment, accuracy = cluster_accuracy(train_targets, predicted)
    #     cluster_centers = kmeans.cluster_centers_[
    #         list(dict(sorted({y: x for x, y in reassignment.items()}.items())).values())]
    #     cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True).to(device)
    #     with torch.no_grad():
    #         # initialise the cluster centers
    #         loss_centroids.state_dict()["centers"].copy_(cluster_centers)

    model = model.train()
    model_class = model_class.train()
    # loss_centroids.train()

    train_losses = []
    target_idx = 0
    train_targets = np.array(train_target)
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      embedding = model.encoder(seq_true).squeeze().to(device)
      y_pred = model_class(embedding)

      loss_ae = criterion(seq_pred, seq_true)

      loss_class = criterion_class(y_pred, torch.tensor([train_targets[target_idx]/n_classes], dtype=torch.float32, device = device))
      target_idx += 1

      # loss_cntrs = loss_centroids(embedding, train_targets[target_idx])

      # loss = 0.5 * loss_ae + 0.5 * loss_cntrs.mean()

      loss_ae.backward()
      loss_class.backward()
      optimizer.step()

      train_losses.append([loss_ae.item(), loss_class.item()])

    val_losses = []
    model = model.eval()
    model_class = model_class.eval()

    val_targets = np.array(val_target)
    target_idx = 0
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        embedding = model.encoder(seq_true).squeeze().to(device)
        y_pred = model_class(embedding)

        loss_ae = criterion(seq_pred, seq_true)
        loss_class = criterion_class(y_pred, torch.tensor([val_targets[target_idx]/n_classes], dtype=torch.float32, device = device))
        target_idx += 1
        val_losses.append([loss_ae.item(), loss_class.item()])

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    train_loss_ae = np.mean(train_losses[:, 0])
    train_loss_class = np.mean(train_losses[:, 1])

    val_loss_ae = np.mean(val_losses[:, 0])
    val_loss_class = np.mean(val_losses[:, 1])

    history['train_ae'].append(train_loss_ae)
    history['train_class'].append(train_loss_class)

    history['val_ae'].append(val_loss_ae)
    history['val_class'].append(val_loss_class)

    if val_loss_ae < best_loss:
      best_loss = val_loss_ae
      best_model_wts = copy.deepcopy(model.state_dict())

    epoch_time = time() - epoch_time
    logger.info(f'Epoch {epoch}: AE TL: {train_loss_ae:.3f}, AE VL:{val_loss_ae:.3f}, CLS TL: {train_loss_class:.3f}, CLS VL:{val_loss_class:.3f}, Epoch time: {epoch_time:.3f}s' )

  model.load_state_dict(best_model_wts)
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
  
def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features, n_seq