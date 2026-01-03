import numpy as np
import pandas as pd
from typing import Tuple
import torch.utils.data as data
from sklearn.model_selection import train_test_split

class AutoRecData(data.Dataset):
    r"""_summary_

    Parameters
    ----------
    data : np.ndarray
        A two-dimensional list is required.
    """

    def __init__(self, data: np.ndarray) -> None:
        super(AutoRecData, self).__init__()

        self.data = data
        self.items = set(data.nonzero()[0])
        self.users = set(data.nonzero()[1])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> None:
        return self.data[index]

