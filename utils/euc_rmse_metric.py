from utils.general import xyxy2xywh
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def filter_preds(preds, stats, confidence_threshold, iou_threshold):
    # Takes in preds and stats and filters to remove if conf low or iou (from stats) low
    filtered_predictions = []
    iou_threshold_map = {0.5 : 0, 0.55 : 1, 0.6 : 2, 0.65 : 3, 0.7 : 4, 0.75 : 5, 0.8 : 6, 0.85 : 7, 0.9 : 8, 0.95 : 9}
    iou_threshold = iou_threshold_map[iou_threshold]
    
    # filter to remove predictions below iou theshold and conf threshold
    for image, stat in zip(preds, stats):
        iou = stat[0][:,iou_threshold] # 0.5 iou
        image = image[iou]
        image = image[image[:, 4] > confidence_threshold]
        image = image[:,:4]
        image = xyxy2xywh(image)
        filtered_predictions.append(image)
    
    return filtered_predictions

def calculate_centroid(tensor_xywh):
    # Assuming the tensor has columns: xcentre, ycentre, height, width
    centroid_x = tensor_xywh[:, 0]
    centroid_y = tensor_xywh[:, 1]
    return torch.stack((centroid_x, centroid_y), dim=1)


def pairwise_euclidean_distance(pred_centroids, actual_centroids):
    # Calculate pairwise Euclidean distance between centroids
    pred_centroids = pred_centroids.unsqueeze(1)  # Add a new dimension to pred_centroids
    actual_centroids = actual_centroids.unsqueeze(0)  # Add a new dimension to actual_centroids
    distances = torch.sqrt(torch.sum((pred_centroids - actual_centroids)**2, dim=2))
    
    # Get the minimum distance along each row
    min_distances, _ = distances.min(dim=1)
    return min_distances

def rmse_tensor(tensor_list):
    
    flattened_tensors = torch.cat([tensor**2 for tensor in tensor_list])
    sum_values = flattened_tensors.sum()

    # Calculate the average
    num_values = flattened_tensors.numel()
    return np.sqrt(sum_values / num_values)


def get_min_euc_rmse_metric(predictions, stats, targets, confidence_threshold, iou_threshold):
    #     get preds above iou and conf threhsold
    filtered_predictions = filter_preds(predictions, stats, confidence_threshold, iou_threshold)
    
    #     for each image, get (min euc distance)**2
    min_euc_dist = []
    for image_preds, image_targets in zip(filtered_predictions, targets):
        pred_centroids = calculate_centroid(image_preds)
        actual_centroids = calculate_centroid(image_targets)
        min_euc_dist.append(pairwise_euclidean_distance(pred_centroids, actual_centroids))
    
    #     get rmse of min euc distances
    return rmse_tensor(min_euc_dist)


def plot_euc_rmse(px, py, save_dir):
    
    # RMSE min euc distance by confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(px, py, linewidth=3, color='blue')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('RMSE min euclidean distance (excl IOU < 0.5) / pixels')
    ax.set_xlim(0, 0.9)
    ax.set_title('RMSE min euclidean distance by confidence curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    

def get_min_euc_rmse_metric_conf_intervals(predictions, stats, targets, iou_threshold, plot = True, save_dir = None):
    confidence_intervals = torch.linspace(0.0, 0.9, 10)  # Generate confidence intervals
    
    min_euc_rmse_metrics = []  # List to store min_euc_rmse_metric for each confidence interval

    for confidence_threshold in confidence_intervals:
        # Get min_euc_rmse_metric for the current confidence_threshold
        min_euc_rmse_metric = get_min_euc_rmse_metric(predictions, stats, targets, confidence_threshold, iou_threshold)
        min_euc_rmse_metrics.append(min_euc_rmse_metric)
    
    if plot:
        plot_euc_rmse(confidence_intervals, min_euc_rmse_metrics, save_dir)

    return min_euc_rmse_metrics
