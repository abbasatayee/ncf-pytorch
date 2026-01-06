import numpy as np
import pandas as pd
import scipy.sparse as sp

def load_all_data(train_rating_path, test_negative_path):    
    # Load training data
    print(f"Loading training data from {train_rating_path}...")
    train_data = pd.read_csv(
        train_rating_path,
        sep='\t',
        header=None,
        names=['user', 'item'],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32}
    )
    
    # Calculate number of users and items
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    
    print(f"✓ Loaded {len(train_data)} training pairs")
    print(f"  - Users: {user_num}")
    print(f"  - Items: {item_num}")
    
    # Convert to list of lists for easier processing
    train_data = train_data.values.tolist()
    
    # Create sparse training matrix (Dictionary of Keys format)
    # This is used to quickly check if a user-item pair exists in training data
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u, i in train_data:
        train_mat[u, i] = 1.0
    
    # Load test data with negative samples
    test_data = []
    with open(test_negative_path, 'r') as fd:
        line = fd.readline()
        while line is not None and line != '':
            # Format: (user, item)\tneg1\tneg2\t...\tneg99
            arr = line.strip().split('\t')
            
            # Parse the positive pair: (user, item)
            # eval() converts string "(123, 456)" to tuple (123, 456)
            positive_pair = eval(arr[0])
            u = positive_pair[0]
            i = positive_pair[1]
            
            # Add the positive pair
            test_data.append([u, i])
            
            # Add all negative items for this user
            for neg_item in arr[1:]:
                if neg_item:  # Skip empty strings
                    test_data.append([u, int(neg_item)])
            
            line = fd.readline()
    
    

    return train_data, test_data, user_num, item_num, train_mat