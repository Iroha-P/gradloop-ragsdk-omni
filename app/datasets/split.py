from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

_DEFAULT_RATIOS = (0.8, 0.1, 0.1)
_DEFAULT_SEED = "dataset-split-v1"


class SplitValidationError(ValueError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[Mapping[str, object], ...]
    dev: tuple[Mapping[str, object], ...]
    test: tuple[Mapping[str, object], ...]
    ratios: tuple[float, float, float] = _DEFAULT_RATIOS
    seed: str = _DEFAULT_SEED
    version: str = field(default="v1", init=False)


def _is_canonical_group_component(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value)
    canonical = " ".join(normalized.split())
    return bool(canonical) and value == canonical


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if (
        not isinstance(ratios, tuple)
        or len(ratios) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or value >= 1
            for value in ratios
        )
        or abs(sum(ratios) - 1.0) > 1e-9
    ):
        raise SplitValidationError("invalid_split_ratios")


def _allocation_counts(
    group_count: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    raw = [group_count * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = group_count - sum(counts)
    remainders = sorted(
        range(3),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1

    if group_count >= 3:
        for empty_index in (1, 2, 0):
            if counts[empty_index] != 0:
                continue
            donor = max(
                (index for index in range(3) if counts[index] > 1),
                key=lambda index: (counts[index], ratios[index]),
                default=-1,
            )
            if donor >= 0:
                counts[donor] -= 1
                counts[empty_index] += 1
    return counts[0], counts[1], counts[2]


def split_by_group(
    samples: Sequence[Mapping[str, object]],
    group_keys: Sequence[str],
    *,
    ratios: tuple[float, float, float] = _DEFAULT_RATIOS,
    seed: str = _DEFAULT_SEED,
) -> DatasetSplit:
    if (
        isinstance(group_keys, (str, bytes))
        or not isinstance(group_keys, Sequence)
        or not group_keys
        or any(not isinstance(key, str) or not _is_canonical_group_component(key) for key in group_keys)
        or len(set(group_keys)) != len(group_keys)
    ):
        raise SplitValidationError("invalid_group_keys")
    _validate_ratios(ratios)
    if not isinstance(seed, str) or not seed.strip():
        raise SplitValidationError("invalid_split_seed")

    grouped: defaultdict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    sample_ids: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise SplitValidationError("invalid_sample")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SplitValidationError("invalid_sample_id")
        if sample_id in sample_ids:
            raise SplitValidationError("duplicate_sample_id")
        sample_ids.add(sample_id)

        values: list[str] = []
        for key in group_keys:
            if key not in sample:
                raise SplitValidationError("missing_group_key")
            value = sample[key]
            if not isinstance(value, str) or not _is_canonical_group_component(value):
                raise SplitValidationError("invalid_group_value")
            values.append(value)
        grouped[tuple(values)].append(sample)

    def group_order(group: tuple[str, ...]) -> tuple[str, str]:
        serialized = json.dumps(group, ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{seed}\0{serialized}".encode()).hexdigest()
        return digest, serialized

    ordered_groups = sorted(grouped, key=group_order)
    train_count, dev_count, _ = _allocation_counts(len(ordered_groups), ratios)
    group_buckets = (
        ordered_groups[:train_count],
        ordered_groups[train_count : train_count + dev_count],
        ordered_groups[train_count + dev_count :],
    )

    def flatten(groups: Sequence[tuple[str, ...]]) -> tuple[Mapping[str, object], ...]:
        output: list[Mapping[str, object]] = []
        for group in groups:
            output.extend(sorted(grouped[group], key=lambda sample: str(sample["sample_id"])))
        return tuple(output)

    train, dev, test = (flatten(groups) for groups in group_buckets)
    assigned_ids = [str(sample["sample_id"]) for bucket in (train, dev, test) for sample in bucket]
    if len(assigned_ids) != len(samples) or len(set(assigned_ids)) != len(assigned_ids):
        raise SplitValidationError("split_contamination")
    return DatasetSplit(
        train=train,
        dev=dev,
        test=test,
        ratios=ratios,
        seed=seed,
    )
