# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Optional
import os
import torch
from fairseq.dataclass.configs import DistributedTrainingConfig
from fairseq.distributed import utils as dist_utils
from typing import Any, Dict, Optional, Set, cast
try:
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.nn.data_parallel import TrainingState
    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False

def free_storage_(data: torch.Tensor):
    if data.storage().size() > 0:
        assert data.storage_offset() == 0
        data.storage().resize_(0)


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    fairseq-specific checkpoint saving/loading logic.

    Args:
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        super().__init__(*args, **kwargs)
        self.use_sharded_state = use_sharded_state

        if dist_utils.get_world_size(group=dist_utils.get_data_parallel_group()) < 4 and \
            "NVIDIA GeForce RTX 3090" in torch.cuda.get_device_name():
            self.alpaca_force_full_precision = False
        else:
            self.alpaca_force_full_precision = True

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        if self.flatten_parameters:
            return self.module.module
        else:
            return self.module

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.use_sharded_state:
            return super().local_state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            if self.rank == 0:
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
            else:
                # We must call state_dict() due to use of communication
                # primitives. But we don't use the result.
                super().state_dict()
                return destination or {}

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        if self.use_sharded_state:
            return super().load_local_state_dict(state_dict, strict=strict)
        else:
            state_dict = dist_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group,
            )
            return super().load_state_dict(state_dict, strict=strict)

    @contextlib.contextmanager
    def summon_full_params(self, recurse: bool = True, volatile: bool = False):
        if recurse:
            with contextlib.ExitStack() as stack:
                # Summon all params for any nested FSDP instances.
                for module in self.modules():
                    if isinstance(module, FullyShardedDataParallel):
                        stack.enter_context(module.summon_full_params(recurse=False, volatile=volatile))
                # Yield to the caller, with full params in all nested instances.
                yield
            # Exiting from the ExitStack will re-shard params.
            return
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            self.assert_state(TrainingState.IDLE)
            # Set the state so that we assert when trying to go into fwd/bwd.
            self.training_state = TrainingState.SUMMON_FULL_PARAMS
            full_tensors = self._rebuild_full_params(force_full_precision=self.alpaca_force_full_precision)
            assert full_tensors is not None
            with contextlib.ExitStack() as stack:
                if self.module.is_flattened:
                    # Update flattened views to point to fully-sized tensors. We
                    # use self.params instead of full_tensors since the
                    # latter may contain padding.
                    stack.enter_context(
                        self.module.unflatten_params(
                            flat_params=[p.data for p in self.params[: self._num_flatten_params]]
                        )
                    )
                try:
                    yield
                finally:
                    stack.close()
                    non_shared_params = self.params
                    # filter out shared params for all but the owner FSDP module.
                    if len(full_tensors) < len(non_shared_params):
                        non_shared_params = self.non_shared_params()
                    assert len(full_tensors) == len(
                        non_shared_params
                    ), f"{len(full_tensors)} vs. {len(non_shared_params)}"
                    for p, (full_tensor, safe_to_free) in zip(non_shared_params, full_tensors):
                        if not volatile:
                            # Copy any changes made to the full params back into
                            # the corresponding local shards.
                            local_shard, _ = self._get_shard(full_tensor)
                            p._fp32_shard.copy_(local_shard.view_as(p._fp32_shard))
                        if safe_to_free:
                            free_storage_(full_tensor)
                    self.has_full_params = False
                    self._use_fp32_param_shard()
                    self.training_state = TrainingState.IDLE


class DummyProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@contextlib.contextmanager
def fsdp_enable_wrap(cfg: DistributedTrainingConfig):
    try:
        from fairscale.nn import enable_wrap
    except ImportError:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install fairscale with: pip install fairscale"
        )
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        group = DummyProcessGroup(rank=0, size=1)
    fsdp_config = {
        "process_group": group,
        "reshard_after_forward": not cfg.no_reshard_after_forward,
        "mixed_precision": cfg.fp16 and not cfg.memory_efficient_fp16,
        "fp32_reduce_scatter": cfg.fp32_reduce_scatter,
        "flatten_parameters": not cfg.not_fsdp_flatten_parameters,
        "cpu_offload": cfg.cpu_offload,
        "compute_dtype": torch.float16 if cfg.fp16 else torch.float32,
        "bucket_cap_mb": cfg.bucket_cap_mb,
        "state_dict_device": torch.device("cpu"),  # reduce GPU mem usage
    }
    with enable_wrap(
        wrapper_cls=FullyShardedDataParallel,
        use_sharded_state=cfg.use_sharded_state,
        **fsdp_config,
    ):
        yield


def fsdp_wrap(module, min_num_params: Optional[int] = None, **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.

    Args:
        module (nn.Module): module to (maybe) wrap
        min_num_params (int, Optional): minimum number of layer params to wrap
    """
    try:
        from fairscale.nn import wrap

        if min_num_params is not None:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params >= min_num_params:
                return wrap(module, **kwargs)
            else:
                return module
        else:
            return wrap(module, **kwargs)
    except ImportError:
        return module
