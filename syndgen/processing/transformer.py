from __future__ import annotations

import pandas as pd

from syndgen.metadata import Metadata
from typing import (
    Any,
    Iterable,
    Optional,
)


class Transform:
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
    ) -> Transform:
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

    def fit(self, *args, **kwargs) -> IdentityTransform:
        """"""
        return self

    def transform(self, data: Any, *args, **kwargs) -> Any:
        """"""
        return data

    def inverse_transform(self, data: Any, *args, **kwargs) -> Any:
        """"""
        return data


class DTypeTransform:
    """"""

    def fit(
        self,
        data: pd.DataFrame,
        metadata: Metadata,
        *,
        discrete_columns: Optional[Iterable[str]] = None,
        continuous_columns: Optional[Iterable[str]] = None,
    ) -> None:
        """"""

        if discrete_columns is None:
            discrete_columns = metadata.detect_discrete_columns()
        
        if continuous_columns is None:
            continuous_columns = metadata.detect_continuous_columns()

        col_metadata = metadata.columns
        
        col_dtype_info = []
        for disc_col in discrete_columns:
            col_dtype_info.append(self._fit_discrete(data[disc_col], col_metadata[disc_col], disc_col))
        
        self._fitted = True
        self._col_dtype_info = col_dtype_info
    
    def _fit_discrete(
        self,
        data: pd.DataFrame,
        col_metadata: Metadata,
        disc_col: str,
    ) -> None:
        """"""

        dtype = col_metadata.get("dtype", None)

        
