"""
MIT License

Copyright (c) 2024 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2024-11-16
Last updated: 2024-11-17
"""

from __future__ import annotations

from typing import Any


class Metadata:
    """"""

    def __init__(self: Metadata) -> None:
        """"""

        self._tables = []
        self._columns = {}
        self._primary_keys = {}
        self._foreign_keys = {}
        self._table_relations = {}

    @classmethod
    def from_dict(cls, _dict: dict[str, Any]) -> Metadata:
        """Create a new ``Metadata`` instance from a dictionary.

        Parameters
        ----------
        _dict : dict
            The dictionary containing all key-value pairs of the metadata.

        Returns
        -------
        Metadata
            The new instance of the ``Metadata`` class.
        
        """

        instance = cls()
        for key, value in _dict.items():
            setattr(instance, key, value)
        return instance 