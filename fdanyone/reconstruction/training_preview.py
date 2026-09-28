"""Publish training metadata and bounded model snapshots atomically.

Standard-library only: imported by both training's isolated Python and tests.
"""
import json
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


class PreviewPublisher:
    def __init__(self, output, status_path, export, *, interval=30, clock=time.monotonic):
        self.output = Path(output)
        self.status_path = Path(status_path)
        self.export = export
        self.interval = interval
        self.clock = clock
        self.last = None

    def publish(self, step):
        now = self.clock()
        if step < 100 or (self.last is not None and now - self.last < self.interval):
            return False
        self.last = now
        relative = f'previews/step-{step:09d}.ftgs.ply'
        path = self.output/relative
        try:
            # The upstream exporter stages the complete file then renames it.
            self.export(path)
            status = json.loads(self.status_path.read_text())
            status.update(preview_model=relative, preview_step=step,
                          preview_seconds=self.clock()-now)
            write_json(self.status_path, status)
        except (OSError, ValueError, RuntimeError) as exc:
            # Optional preview failure must not discard a useful training run.
            print(f'[Preview] Could not publish step {step}: {exc}', flush=True)
            return False
        # Retain a previous immutable snapshot for downloads already in flight.
        for old in sorted(path.parent.glob('step-*.ftgs.ply'))[:-2]:
            try:
                old.unlink()
            except OSError:
                pass
        return True


def clear_previews(output):
    """Called after the final model and completed status are published."""
    directory = Path(output)/'previews'
    for path in directory.glob('step-*.ftgs.ply'):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass
