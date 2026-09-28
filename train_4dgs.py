"""Train an existing calibrated result without regenerating its videos."""
from fdanyone.reconstruction import train_result

if __name__ == '__main__':
    import sys
    from fire import Fire
    from fdanyone.errors import FourDAnyoneError
    try:
        Fire(train_result)
    except FourDAnyoneError as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
