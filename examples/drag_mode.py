"""Run the B601-aligned drag controller on Piper (the only drag implementation).

    python examples/drag_mode.py --help

Uses the patched Piper SDK environment when necessary.
"""
import os
import sys


def main():
    piper_python = os.path.expanduser('~/miniforge3/envs/piperctl/bin/python')
    try:
        from piperx_teleop import require_patched_sdk
        require_patched_sdk()
    except (RuntimeError, ImportError):
        if os.path.exists(piper_python) and os.path.realpath(sys.executable) != os.path.realpath(piper_python):
            os.execv(piper_python, [piper_python] + sys.argv)
        raise
    from b601_drag import main as run
    run()


if __name__ == '__main__':
    main()
