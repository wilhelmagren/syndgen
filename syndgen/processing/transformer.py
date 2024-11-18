from __future__ import annotations

import pandas as pd

from typing import Any


class Transformer:
    """"""

    def __init__(
        self,
        *,
        max_bgm_clusters: int = 10,
        weight_threshold: float = 5e-3,
    ) -> None:
        """"""

        self._max_bgm_clusters = max_bgm_clusters
        self._weight_threshold = weight_threshold
    
    def fit(
        self,
        data: pd.DataFrame,
    ) -> Transformer:
        """"""

        n_output_dims = 0
        column_info = {"activation": [], "transform": []}

        if len(data) <= self._max_bgm_clusters:
            self._max_bgm_clusters = max(len(data) - 1, 1)
        
        for column in data.columns:
            pass

        self._fitted = True


class IdentityTransform:
    """"""

    def fit(self, *args, **kwargs) -> None:
        """"""

    def transform(self, data: Any, *args, **kwargs) -> Any:
        """"""
        return data

    def inverse_transform(self, data: Any, *args, **kwargs) -> Any:
        """"""
        return data
