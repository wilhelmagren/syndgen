import pandas as pd
from syndgen.metadata import Metadata

class Dataset:
    """"""
    
    def __init__(
        self,
        data: pd.PataFrame,
        metadata: Metadata, 
    ) -> None:
        """"""

        self._data = data
        self._metadata = metadata
    
    def __len__(self) -> int:
        """"""
        return self._data.shape[0]
    
    def fit(self) -> None:
        """"""
        raise NotImplementedError