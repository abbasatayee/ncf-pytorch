import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

class PreProcessor():
    def __init__(self) -> None:
        pass

    def preprocess_ml1m_data(self, data: pd.DataFrame, test_size: float = 0.1, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, int, int]:
         # Handle different column name formats
        if "user_id" in data.columns and "item_id" in data.columns:
            # Create a copy to avoid modifying the original
            data = data.copy()
            data["user"] = data["user_id"]
            data["item"] = data["item_id"]
        elif "user" not in data.columns or "item" not in data.columns:
            raise ValueError(
                "Data must contain either ('user_id', 'item_id') or ('user', 'item') columns"
            )
        
        # Ensure we have rating column
        if "rating" not in data.columns:
            raise ValueError("Data must contain a 'rating' column")
        
        # Get number of unique users and items (before remapping)
        # Add 1 because IDs might be 0-indexed, and we need max+1 for array size
        num_users = int(data["user"].max() + 1)
        num_items = int(data["item"].max() + 1)
        
        # Remap user and item IDs to be contiguous (0-indexed) if needed
        # Check if IDs are already 0-indexed and contiguous
        unique_users = sorted(data["user"].unique())
        unique_items = sorted(data["item"].unique())
        
        # If not contiguous, remap
        if len(unique_users) != num_users or unique_users != list(range(num_users)):
            user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
            data["user"] = data["user"].map(user_map)
            num_users = len(unique_users)
        
        if len(unique_items) != num_items or unique_items != list(range(num_items)):
            item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
            data["item"] = data["item"].map(item_map)
            num_items = len(unique_items)
        
        # Select only the columns we need for train/test split
        data_for_split = data[["user", "item", "rating"]].copy()
        
        # Split into train and test
        train, test = train_test_split(
            data_for_split.values, test_size=test_size, random_state=random_state
        )

        # Create user-item rating matrices
        train_mat = np.zeros((num_users, num_items), dtype=np.float32)
        test_mat = np.zeros((num_users, num_items), dtype=np.float32)

        # Fill train matrix
        for user, item, rating in train:
            train_mat[int(user), int(item)] = float(rating)
        
        # Fill test matrix
        for user, item, rating in test:
            test_mat[int(user), int(item)] = float(rating)

        return train_mat, test_mat, num_users, num_items