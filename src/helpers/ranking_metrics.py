"""
Helper functions for ranking metrics.
Hit Rate (HR) and Normalized Discounted Cumulative Gain (NDCG) are two commonly used metrics for evaluating the performance of recommendation systems.
"""
import numpy as np

def hit(gt_item, pred_items):
    """
    Calculate Hit Rate for a single test case.
    
    Hit Rate is 1 if the ground truth item is in the predicted top-K items,
    otherwise 0.
    
    Parameters:
    - gt_item: Ground truth item ID (the item user actually interacted with)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - 1 if gt_item is in pred_items, 0 otherwise
    """
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a single test case.
    
    NDCG measures ranking quality by:
    1. Giving more weight to items ranked higher (position matters)
    2. Using logarithmic discounting (relevance decreases with position)
    
    Formula: NDCG = 1 / log2(position + 2)
    - Position 0 (top): 1 / log2(2) = 1.0
    - Position 1: 1 / log2(3) ≈ 0.63
    - Position 2: 1 / log2(4) = 0.5
    - Position 9: 1 / log2(11) ≈ 0.29
    
    Parameters:
    - gt_item: Ground truth item ID (the item user actually interacted with)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - NDCG score (0.0 to 1.0) if gt_item is in pred_items
    - 0.0 if gt_item is not in pred_items
    """
    if gt_item in pred_items:
        # Find the position (index) of the ground truth item
        index = pred_items.index(gt_item)
        # Calculate NDCG: 1 / log2(position + 2)
        # +2 because: position 0 should give 1/log2(2) = 1.0
        return np.reciprocal(np.log2(index + 2))
    return 0.0
