"""Shared guards for projection training loops."""
from __future__ import annotations


def ensure_full_batch(sample_count: int, batch_size: int) -> None:
    """Fail fast when drop_last=True would yield zero batches."""
    if batch_size < 2:
        raise ValueError("batch_size 必须 >= 2")
    if sample_count < batch_size:
        raise ValueError(
            f"训练样本只有 {sample_count} 条，小于 batch_size={batch_size}；"
            "drop_last=True 会产生空 batch，模型不会更新"
        )
