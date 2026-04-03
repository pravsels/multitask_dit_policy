from __future__ import annotations

import importlib
import importlib.util


def _load_sampler_module():
    spec = importlib.util.find_spec("multitask_dit_policy.utils.distributed_sampler")
    assert spec is not None, "expected distributed sampler module to exist"
    return importlib.import_module("multitask_dit_policy.utils.distributed_sampler")


def test_distributed_index_sampler_shards_indices_without_overlap():
    sampler_module = _load_sampler_module()
    assert hasattr(sampler_module, "DistributedIndexSampler")

    sampler0 = sampler_module.DistributedIndexSampler(
        indices=list(range(8)),
        num_replicas=2,
        rank=0,
        shuffle=False,
        drop_last=False,
    )
    sampler1 = sampler_module.DistributedIndexSampler(
        indices=list(range(8)),
        num_replicas=2,
        rank=1,
        shuffle=False,
        drop_last=False,
    )

    assert list(sampler0) == [0, 2, 4, 6]
    assert list(sampler1) == [1, 3, 5, 7]


def test_distributed_index_sampler_changes_order_across_epochs():
    sampler_module = _load_sampler_module()
    assert hasattr(sampler_module, "DistributedIndexSampler")

    sampler = sampler_module.DistributedIndexSampler(
        indices=list(range(10)),
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=False,
        seed=7,
    )

    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert epoch0 != epoch1
