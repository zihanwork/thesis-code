from __future__ import annotations


class ETPlanAdapter:
    def load(self, path: str):
        raise NotImplementedError("ET-Plan-Bench conversion requires benchmark task metadata.")
