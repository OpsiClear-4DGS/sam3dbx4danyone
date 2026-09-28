"""Run one video job per GPU, automatically using all visible GPUs."""
import sys
import signal
from fdanyone.errors import FourDAnyoneError
from fdanyone.streams import batch_inference

if __name__ == '__main__':
    from fire import Fire
    def cancel(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancel)
    try:
        Fire(batch_inference)
    except KeyboardInterrupt:
        print('Cancelled; completed outputs and the queue report are preserved.', file=sys.stderr)
        raise SystemExit(130) from None
    except FourDAnyoneError as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
