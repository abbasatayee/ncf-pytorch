import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from itertools import product


def get_metrics(
    model: nn.Module, train_set=data.Dataset, test_set=data.Dataset, device=None
) -> np.float32:
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get matrix from torch Dataset
    test_mat = torch.Tensor(test_set.data).to(device)
    test_mask = (test_mat > 0).to(device)

    # Reconstruct the test matrix
    reconstruction = model(test_mat)

    # Get unseen users and items
    unseen_users = test_set.users - train_set.users
    unseen_items = test_set.users - train_set.items

    # Use a default rating of 3 for test users or
    # items without training observations.
    for item, user in product(unseen_items, unseen_users):
        if test_mask[user, item]:
            reconstruction[user, item] = 3

    return masked_rmse(actual=test_mat, pred=reconstruction, mask=test_mask)


def masked_rmse(
    actual: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> np.float32:
    mse = ((pred - actual) * mask).pow(2).sum() / mask.sum()

    return np.sqrt(mse.detach().cpu().numpy())


def hit(gt_items, pred_items):
    """
    Calculate Hit Rate for a single user.
    
    Hit Rate is 1 if any ground truth item is in the predicted top-K items,
    otherwise 0.
    
    Parameters:
    - gt_items: Set or list of ground truth item IDs (items user actually interacted with in test set)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - 1 if any gt_item is in pred_items, 0 otherwise
    """
    if len(gt_items) == 0:
        return 0.0
    return 1.0 if any(item in pred_items for item in gt_items) else 0.0


def ndcg(gt_items, pred_items):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a single user.
    
    NDCG measures ranking quality by:
    1. Giving more weight to items ranked higher (position matters)
    2. Using logarithmic discounting (relevance decreases with position)
    
    Formula: NDCG = sum(1 / log2(position + 2)) for all gt_items in pred_items
    
    Parameters:
    - gt_items: Set or list of ground truth item IDs (items user actually interacted with in test set)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - NDCG score (0.0 to 1.0) normalized by the number of ground truth items
    """
    if len(gt_items) == 0:
        return 0.0
    
    # Calculate DCG: sum of 1/log2(position + 2) for each ground truth item found
    dcg = 0.0
    for idx, item in enumerate(pred_items):
        if item in gt_items:
            dcg += 1.0 / np.log2(idx + 2)
    
    # Calculate IDCG (Ideal DCG): if all ground truth items were at the top
    # This normalizes the score to [0, 1]
    num_gt = len(gt_items)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, len(pred_items))))
    
    # Normalize: NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_ranking_metrics(
    model: nn.Module, 
    train_set: data.Dataset, 
    test_set: data.Dataset, 
    top_k: int = 10,
    device=None
) -> tuple:
    """
    Calculate HR@K and NDCG@K metrics for AutoRec model.
    
    For each user:
    1. Get model predictions for all items
    2. Mask out items seen in training (to avoid recommending already seen items)
    3. Get top-K items with highest predicted ratings
    4. Check if test items are in top-K (HR@K)
    5. Calculate NDCG@K based on positions of test items
    
    Parameters:
    - model: Trained AutoRec model
    - train_set: Training dataset (to mask out seen items)
    - test_set: Test dataset (to get ground truth items)
    - top_k: Number of top items to consider (default: 10)
    - device: Device to run on (default: auto-detect)
    
    Returns:
    - mean_HR: Average Hit Rate across all users
    - mean_NDCG: Average NDCG across all users
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Get matrices
    train_mat = torch.Tensor(train_set.data).to(device)
    test_mat = torch.Tensor(test_set.data).to(device)
    
    # Get training mask (items seen in training)
    train_mask = (train_mat > 0).to(device)
    
    # Get predictions for all users
    with torch.no_grad():
        # Use training matrix as input to predict ratings for all items
        # This simulates: given what user has seen, what would they rate for unseen items?
        predictions = model(train_mat)
        
        # Mask out items seen in training (set to very low value)
        # This ensures we only recommend unseen items
        predictions = predictions * (~train_mask).float() - train_mask.float() * 1e10
    
    # Move to CPU for numpy operations
    predictions = predictions.cpu().numpy()
    test_mat_np = test_mat.cpu().numpy()
    train_mat_np = train_mat.cpu().numpy()
    
    HR_list = []
    NDCG_list = []
    
    # For each user, calculate metrics
    num_users = predictions.shape[0]
    for user_id in range(num_users):
        # Get ground truth items for this user (items rated in test set)
        test_items = set(np.where(test_mat_np[user_id] > 0)[0])
        
        # Skip if user has no test items
        if len(test_items) == 0:
            continue
        
        # Get top-K items for this user
        user_predictions = predictions[user_id]
        top_k_indices = np.argsort(user_predictions)[-top_k:][::-1]  # Get top-K, descending order
        top_k_items = top_k_indices.tolist()
        
        # Calculate metrics
        HR_list.append(hit(test_items, top_k_items))
        NDCG_list.append(ndcg(test_items, top_k_items))
    
    # Calculate average metrics
    if len(HR_list) == 0:
        return 0.0, 0.0
    
    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)
    
    return mean_HR, mean_NDCG