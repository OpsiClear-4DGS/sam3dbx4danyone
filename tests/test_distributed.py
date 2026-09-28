from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fdanyone.model.distributed import _WorkerState, _configure_nccl_environment


class DistributedStagingTests(unittest.TestCase):
    def _state(self, latents: torch.Tensor | None, *, rank: int = 0) -> _WorkerState:
        dtype = torch.float32 if latents is None else latents.dtype
        width = 2 if latents is None else latents.shape[1]
        return _WorkerState(
            rank=rank,
            request=SimpleNamespace(devices=("cpu:0", "cpu:1", "cpu:2")),
            denoiser=None,
            device="cpu",
            device_index=0,
            source=torch.empty(0),
            context=torch.empty(0),
            null_pose_feature=torch.empty(0),
            pose_features=torch.empty(0),
            latents=latents,
            pose_feature_batch=torch.empty(2, 1),
            accumulation_dtype=dtype,
            latent_tail=(width,),
        )

    def test_scatter_reuses_one_remote_staging_buffer(self) -> None:
        latents = torch.arange(12, dtype=torch.float32).view(6, 2)
        state = self._state(latents)
        wave = ((4, 0), (1, 5), (3, 2))
        broadcasts: list[tuple[int, int, torch.Tensor]] = []

        def capture(tensor, *, src):
            broadcasts.append((src, id(tensor), tensor.clone()))

        with patch("torch.distributed.broadcast", side_effect=capture):
            local = state._scatter_latents(wave)

        torch.testing.assert_close(local, latents[list(wave[0])])
        self.assertEqual([src for src, _, _ in broadcasts], [0, 0, 0])
        self.assertEqual(broadcasts[1][1], broadcasts[2][1])
        self.assertNotEqual(broadcasts[0][1], broadcasts[1][1])
        torch.testing.assert_close(broadcasts[0][2], latents[list(wave[0])])
        torch.testing.assert_close(broadcasts[1][2], latents[list(wave[1])])
        torch.testing.assert_close(broadcasts[2][2], latents[list(wave[2])])

    def test_gather_reuses_one_remote_staging_buffer_and_commits_routes(self) -> None:
        state = self._state(torch.zeros(6, 2, dtype=torch.float32))
        wave = ((4, 0), (1, 5), (3, 2))
        results = {
            1: torch.tensor([[10.0, 11.0], [50.0, 51.0]]),
            2: torch.tensor([[30.0, 31.0], [20.0, 21.0]]),
        }
        broadcast_ids: list[int] = []

        def broadcast(tensor, *, src):
            broadcast_ids.append(id(tensor))
            if src > 0:
                tensor.copy_(results[src])

        local = torch.tensor([[40.0, 41.0], [0.0, 1.0]])
        with patch("torch.distributed.broadcast", side_effect=broadcast):
            state._gather_and_commit(local, wave)

        self.assertNotEqual(broadcast_ids[0], broadcast_ids[1])
        self.assertEqual(broadcast_ids[1], broadcast_ids[2])
        expected = torch.tensor(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        )
        torch.testing.assert_close(state.latents, expected)

    def test_remote_rank_preserves_its_local_result_across_ordered_broadcasts(self) -> None:
        state = self._state(None, rank=1)
        wave = ((4, 0), (1, 5), (3, 2))
        local = torch.tensor([[10.0, 11.0], [50.0, 51.0]])

        def broadcast(tensor, *, src):
            if src != 1:
                tensor.fill_(float(src))

        with patch("torch.distributed.broadcast", side_effect=broadcast):
            state._gather_and_commit(local, wave)

        torch.testing.assert_close(local, torch.tensor([[10.0, 11.0], [50.0, 51.0]]))

    def test_remote_rank_retains_only_its_scattered_group(self) -> None:
        state = self._state(None, rank=1)
        wave = ((4, 0), (1, 5), (3, 2))
        groups = (
            torch.tensor([[40.0, 41.0], [0.0, 1.0]]),
            torch.tensor([[10.0, 11.0], [50.0, 51.0]]),
            torch.tensor([[30.0, 31.0], [20.0, 21.0]]),
        )
        broadcast_index = 0

        def broadcast(tensor, *, src):
            nonlocal broadcast_index
            self.assertEqual(src, 0)
            tensor.copy_(groups[broadcast_index])
            broadcast_index += 1

        with patch("torch.distributed.broadcast", side_effect=broadcast):
            local = state._scatter_latents(wave)

        torch.testing.assert_close(local, groups[1])


class NcclEnvironmentTests(unittest.TestCase):
    def test_safe_single_node_defaults_are_added(self) -> None:
        environment: dict[str, str] = {}
        _configure_nccl_environment(environment)

        self.assertEqual(environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertEqual(environment["CUDA_DEVICE_MAX_CONNECTIONS"], "1")
        self.assertEqual(environment["NCCL_P2P_DISABLE"], "1")

    def test_validated_caller_override_is_preserved(self) -> None:
        environment = {"NCCL_P2P_DISABLE": "0"}
        _configure_nccl_environment(environment)

        self.assertEqual(environment["NCCL_P2P_DISABLE"], "0")


if __name__ == "__main__":
    unittest.main()
