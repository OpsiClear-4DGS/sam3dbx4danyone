from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from fdanyone.errors import AssetError, FourDAnyoneError
from fdanyone.model.inference import _target_routes, _turbo_target_routes
from fdanyone.model.turbo_lora import fuse_turbo_lora, validate_turbo_base_metadata
from fdanyone.views import resolve_view_plan


class TurboTests(unittest.TestCase):
    def test_low_rank_and_bias_fusion_matches_fp32_reference(self):
        model = torch.nn.Linear(3, 2).to(torch.bfloat16)
        weight = model.weight.detach().float().clone()
        bias = model.bias.detach().float().clone()
        down = torch.tensor([[.2, -.1, .3]], dtype=torch.float16)
        up = torch.tensor([[.4], [-.2]], dtype=torch.float16)
        delta = torch.tensor([.1, -.3], dtype=torch.float16)
        # Prefix a nested layer exactly as the real adapter names its targets.
        wrapper = torch.nn.ModuleDict({"layer": model})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({
                "diffusion_model.layer.lora_down.weight": down,
                "diffusion_model.layer.lora_up.weight": up,
                "diffusion_model.layer.diff_b": delta,
            }, str(path))
            fuse_turbo_lora(wrapper, path)
            fused = {key: value.clone() for key, value in wrapper.state_dict().items()}
            fuse_turbo_lora(wrapper, path)
            for key, value in wrapper.state_dict().items():
                torch.testing.assert_close(value, fused[key], rtol=0, atol=0)
            another = Path(directory) / 'different.safetensors'
            another.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(AssetError, 'already fused'):
                fuse_turbo_lora(wrapper, another)
        torch.testing.assert_close(model.weight, (weight + up.float() @ down.float()).bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(model.bias, (bias + delta.float()).bfloat16(), rtol=0, atol=0)

    def test_previously_fused_checkpoint_is_rejected(self):
        with self.assertRaises(AssetError):
            validate_turbo_base_metadata({"4danyone.turbo_lora_sha256": "already fused"})

    def test_upstream_four_step_routing_and_base_unchanged(self):
        plan = resolve_view_plan(views_per_layer=6)
        routes = _turbo_target_routes(plan)
        self.assertEqual([step[0][0] for step in routes], [0, 2, 4, 0])
        for step in routes:
            self.assertEqual(sorted(step[0]), list(range(6)))
        self.assertEqual(len(_target_routes(plan)), 24)

    def test_preset_routes_cover_each_camera_and_cross_group_boundaries(self):
        from fdanyone.model.distributed import validate_routes
        for count in (12, 18, 24, 36):
            with self.subTest(count=count):
                plan = resolve_view_plan(views_per_layer=count, enable_rcp=False)
                routes = _turbo_target_routes(plan)
                validate_routes(routes, count)
                self.assertEqual(len(routes), 4)
                self.assertTrue(all(len(step) == count // 6 for step in routes))
                self.assertEqual(routes[0][0], (0, 1, 2, 3, 4, 5))
                self.assertEqual(routes[1][0], (2, 3, 4, 5, 6, 7))
                self.assertEqual(routes[2][-1], (count-2, count-1, 0, 1, 2, 3))
                self.assertEqual(routes[3], routes[0])
                frozen = _turbo_target_routes(resolve_view_plan(
                    views_per_layer=count, enable_rcp=False, enable_tcr=False))
                self.assertTrue(all(step == routes[0] for step in frozen))

    def test_elevation_presets_cover_every_camera_in_fixed_size_groups(self):
        from fdanyone.model.distributed import validate_routes
        for total, rings in ((6,1),(8,1),(12,1),(16,1),(16,2),
                             (18,1),(18,2),(24,2),(24,3),(36,3)):
            pitches = {1:[15], 2:[0,30], 3:[-15,15,45]}[rings]
            for turbo in (True, False):
                with self.subTest(total=total, rings=rings, turbo=turbo):
                    plan = resolve_view_plan(views_per_layer=total//rings,
                                             layer_pitches=pitches, enable_rcp=not turbo)
                    self.assertEqual(plan.num_target_views, total)
                    self.assertEqual(plan, type(plan).from_dict(plan.to_dict()))
                    routes = _turbo_target_routes(plan) if turbo else _target_routes(plan)
                    validate_routes(routes, total)
                    self.assertEqual(len(routes), 4 if turbo else 24)
                    self.assertTrue(all(len(step)==plan.num_groups for step in routes))
                    self.assertTrue(all(len(group)==plan.views_per_group
                                        for step in routes for group in step))
                    for layer, pitch in enumerate(pitches):
                        views = [v for v in plan.target_views if v.layer_index==layer]
                        self.assertEqual(len(views), total//rings)
                        self.assertTrue(all(v.pitch==pitch for v in views))
                        self.assertEqual([v.yaw for v in views],
                                         [i*360/(total//rings) for i in range(total//rings)])
                    if (total, rings)==(18,2):
                        self.assertIsNone(plan.groups_per_layer)
                        self.assertTrue(any(len({i//9 for i in group})==2
                                            for group in routes[0]))
                    if turbo:
                        self.assertEqual(routes[-1], routes[0])

    def test_unsupported_layout_fails_before_inference(self):
        for plan in (resolve_view_plan(views_per_layer=24),
                     resolve_view_plan(views_per_layer=20, enable_rcp=False),
                     resolve_view_plan(views_per_layer=6, layer_pitches=[-15, 0, 15, 30], enable_rcp=False)):
            with self.assertRaises(FourDAnyoneError):
                _turbo_target_routes(plan)


if __name__ == "__main__":
    unittest.main()
