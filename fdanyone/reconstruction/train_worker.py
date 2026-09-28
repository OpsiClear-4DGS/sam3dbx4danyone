"""Subprocess bridge, executed with FreeTimeGsVanilla's Python environment.

Keep imports independent of fdanyone: training uses Python 3.12 / NumPy 1,
whereas generation uses Python 3.11 / NumPy 2.
"""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
if __package__:
    from .training_preview import PreviewPublisher, write_json
else:
    from training_preview import PreviewPublisher, write_json


def validate_dataset_manifest(manifest):
    if manifest.get('format_version') != 2 or manifest.get('alpha_mode') != 'transparent':
        raise ValueError('Training requires preserved soft alpha. Regenerate this dataset from the '
                         'original videos; legacy black-composited RGB has no recoverable mask.')


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    repo = Path(request['repo'])
    sys.path[:0] = [str(repo/'src'), str(repo)]
    import numpy as np
    import torch
    import simple_trainer_freetime_4d_pure_relocation as trainer
    from datasets.FreeTime_dataset import FreeTimeDataset, FreeTimeParser
    from datasets.normalize import similarity_from_cameras, transform_cameras, transform_points
    from export_ftgs_ply import save_ftgs_ply
    from gsplat.strategy import DefaultStrategy

    if request.get('check'):
        if 'alpha_mode' not in trainer.Config.__dataclass_fields__ or 'lambda_alpha' not in trainer.Config.__dataclass_fields__:
            raise RuntimeError('Update FreeTimeGsVanilla: this bridge requires transparent RGBA training support.')
        if not torch.cuda.is_available():
            raise RuntimeError('FreeTimeGS requires a visible CUDA GPU.')
        print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__}))
        return

    class UprightParser(FreeTimeParser):
        """Normalize scale/center while retaining our canonical human axes.

        The generic parser's PCA can turn a standing person sideways. Preserve
        Y up for a predictable exported scene and browser orbit controls.
        """
        def __init__(self, *args, normalize=True, **kwargs):
            super().__init__(*args, normalize=False, **kwargs)
            if normalize:
                transform = similarity_from_cameras(self.camtoworlds)
                self.camtoworlds = transform_cameras(transform, self.camtoworlds)
                self.points = transform_points(transform, self.points)
                self.scene_scale *= float(transform[0, 0])
                self.transform = transform
    # The runner constructs its parser internally. Keep this axis adaptation
    # local to the isolated adapter process.
    trainer.FreeTimeParser = UprightParser

    output = Path(request['output'])
    dataset = Path(request['dataset'])
    manifest = json.loads((dataset/'manifest.json').read_text())
    validate_dataset_manifest(manifest)
    steps = request['steps']
    # Relocation preset adapted to surface initialization: upstream's tiny
    # initial scales are intended for much denser triangulated point clouds.
    cfg = trainer.Config(
        data_dir=str(dataset), result_dir=str(output), init_npz_path=str(dataset/'init.npz'),
        start_frame=0, end_frame=manifest['frames'], max_steps=steps,
        save_steps=[steps], eval_steps=[], no_sampling=True, use_keyframe_sampling=True,
        init_scale=1.0, init_opacity=.5, velocity_lr_start=5e-3, velocity_lr_end=1e-4,
        densification_start_step=100, relocation_stop_iter=int(.9*steps),
        relocation_max_ratio=.10, packed=True, lambda_4d_reg=1e-4, lambda_duration_reg=1e-3,
        alpha_mode='transparent', lambda_alpha=.1, random_bkgd=True,
        strategy=DefaultStrategy(verbose=False, refine_start_iter=steps+1,
                                 refine_stop_iter=steps+1, reset_every=steps+1),
        render_traj_n_frames=manifest['frames'], render_traj_time_frames=manifest['frames'],
        render_traj_fps=max(1, round(float(Fraction(manifest['fps'])))),
        export_ply=False, disable_viewer=True,
    )
    runner = trainer.FreeTime4DRunner(0, 0, 1, cfg)
    # Reconstruction uses every generated camera, including the front view.
    # Training loss is not a held-out or real-world reconstruction benchmark.
    runner.trainset = FreeTimeDataset(runner.parser, split='train', test_set=[], alpha_mode=cfg.alpha_mode)
    transform = np.asarray(runner.parser.transform)
    write_json(output/'normalization.json', dict(
        world_to_training=transform.tolist(), training_to_world=np.linalg.inv(transform).tolist(),
        cameras=runner.parser.camtoworlds.tolist(),
        target=runner.parser.points.mean(axis=0).tolist(),
        world_up=(transform[:3, :3] @ np.array([0, 1, 0])).tolist()))

    original_writer = runner.writer
    publisher = PreviewPublisher(output, request['status'],
        lambda path: save_ftgs_ply(path, runner.splats,
            use_velocity=cfg.use_velocity, n_frames=manifest['frames']))
    class ProgressWriter:
        def __getattr__(self, name):
            return getattr(original_writer, name)

        def add_scalar(self, tag, value, step, *args, **kwargs):
            original_writer.add_scalar(tag, value, step, *args, **kwargs)
            if tag == 'loss/total':
                if not math.isfinite(float(value)):
                    raise RuntimeError('FreeTimeGS produced a non-finite training loss.')
                status_path = Path(request['status'])
                status = json.loads(status_path.read_text())
                status.update(stage='training', step=step+1, loss=float(value))
                write_json(status_path, status)
                if step+1 < steps:
                    publisher.publish(step+1)
    runner.writer = ProgressWriter()
    try:
        runner.train()
        # Upstream catches optional export failures. Perform the required model
        # export here so a failed export cannot be reported as a completed job.
        save_ftgs_ply(output/'scene.ftgs.ply', runner.splats,
                      use_velocity=cfg.use_velocity, n_frames=manifest['frames'])
    finally:
        original_writer.close()


if __name__ == '__main__':
    main()
