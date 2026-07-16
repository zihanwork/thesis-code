from __future__ import annotations


class RetryBudget:
    def __init__(self, max_retries: int) -> None:
        self.max_retries = max_retries

    def exhausted(self, retry_index: int) -> bool:
        return retry_index >= self.max_retries
