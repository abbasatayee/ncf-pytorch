import os
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _load_and_filter_ratings(ratings_file):
    """Load ratings from file and filter for positive interactions (>= 4)."""
    ratings = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        dtype={'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.float32, 'Timestamp': np.int32}
    )
    
    # Filter ratings >= 4 (positive interactions)
    # In recommendation systems, we typically treat ratings >= 4 as positive
    positive_ratings = ratings[ratings['Rating'] >= 4].copy()
    
    return positive_ratings


def _remap_ids(positive_ratings):
    """Remap user and item IDs to be contiguous (0-indexed)."""
    unique_users = sorted(positive_ratings['UserID'].unique())
    unique_items = sorted(positive_ratings['MovieID'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    positive_ratings['user'] = positive_ratings['UserID'].map(user_map)
    positive_ratings['item'] = positive_ratings['MovieID'].map(item_map)
    
    user_num = len(unique_users)
    item_num = len(unique_items)
    
    return positive_ratings, user_num, item_num


def _split_train_test(user_item_pairs, test_ratio=0.2):
    """Split user-item pairs into train and test sets."""
    np.random.seed(42)  # For reproducibility
    n_total = len(user_item_pairs)
    n_test = int(n_total * test_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_pairs = user_item_pairs[train_indices]
    test_pairs = user_item_pairs[test_indices]
    
    return train_pairs, test_pairs


def _save_training_data(train_pairs, data_dir):
    """Save training data to file."""
    train_file = os.path.join(data_dir, 'ml-1m.train.rating')
    train_df = pd.DataFrame(train_pairs, columns=['user', 'item'])
    train_df.to_csv(train_file, sep='\t', header=False, index=False)
    return train_file


def _create_training_matrix(train_pairs, user_num, item_num):
    """Create sparse training matrix for negative sampling."""
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u, i in train_pairs:
        train_mat[u, i] = 1.0
    print(f"✓ Training matrix created: {train_mat.nnz} interactions")
    return train_mat


def _generate_negative_samples_for_user(u, i, train_mat, item_num, test_negatives):
    """Generate negative samples for a single user-item pair."""
    negatives = []
    attempts = 0
    max_attempts = test_negatives * 10  # Safety limit
    
    # Sample negative items (items not in training set for this user)
    while len(negatives) < test_negatives and attempts < max_attempts:
        neg_item = np.random.randint(item_num)
        # Make sure it's not in training set for this user
        if (u, neg_item) not in train_mat:
            negatives.append(neg_item)
        attempts += 1
    
    # If we couldn't find enough negatives, pad with random items
    while len(negatives) < test_negatives:
        neg_item = np.random.randint(item_num)
        if neg_item not in negatives:
            negatives.append(neg_item)
    
    return negatives


def _save_test_negative_samples(test_pairs, train_mat, item_num, data_dir, test_negatives=99):
    """Generate and save test negative samples in NCF format."""
    test_negative_file = os.path.join(data_dir, 'ml-1m.test.negative')
    
    with open(test_negative_file, 'w') as f:
        for u, i in test_pairs:
            negatives = _generate_negative_samples_for_user(u, i, train_mat, item_num, test_negatives)
            
            # Write in NCF format: (user, item)\tneg1\tneg2\t...\tneg99
            line = f"({u}, {i})" + "\t" + "\t".join(map(str, negatives)) + "\n"
            f.write(line)
    
    return test_negative_file


def _save_test_data(test_pairs, data_dir):
    """Save test data (for reference, though NCF mainly uses test.negative)."""
    test_file = os.path.join(data_dir, 'ml-1m.test.rating')
    test_df = pd.DataFrame(test_pairs, columns=['user', 'item'])
    test_df.to_csv(test_file, sep='\t', header=False, index=False)
    return test_file


def preprocess_ml1m_to_ncf_format(ratings_file, data_dir, test_ratio=0.2, test_negatives=99):
    """Preprocess ML-1M dataset to NCF format.
    
    Args:
        ratings_file: Path to the ratings file
        data_dir: Directory to save output files
        test_ratio: Ratio of test data (default: 0.2)
        test_negatives: Number of negative samples per test item (default: 99)
    
    Returns:
        Tuple of (train_file, test_file, test_negative_file, user_num, item_num, train_mat)
    """
    # Load and filter ratings
    positive_ratings = _load_and_filter_ratings(ratings_file)
    
    # Remap IDs to be contiguous
    positive_ratings, user_num, item_num = _remap_ids(positive_ratings)
    
    # Create user-item pairs
    user_item_pairs = positive_ratings[['user', 'item']].values
    
    # Split into train and test sets
    train_pairs, test_pairs = _split_train_test(user_item_pairs, test_ratio)
    
    # Save training data
    train_file = _save_training_data(train_pairs, data_dir)
    
    # Create training matrix (for negative sampling)
    train_mat = _create_training_matrix(train_pairs, user_num, item_num)
    
    # Generate and save test negative samples
    test_negative_file = _save_test_negative_samples(
        test_pairs, train_mat, item_num, data_dir, test_negatives
    )
    
    # Save test data (for reference, though NCF mainly uses test.negative)
    test_file = _save_test_data(test_pairs, data_dir)
    
    return train_file, test_file, test_negative_file, user_num, item_num, train_mat