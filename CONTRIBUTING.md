# Contributing

Wan Video Studio is a Windows-first, offline media pipeline. Keep changes local-first: no telemetry, cloud dependency, remote API requirement, or automatic model upload.

## Development

1. Create a Python 3.13 virtual environment.
2. Install `requirements.txt`.
3. Set any non-default paths using the `WAN_*` environment variables documented in `README.md`.
4. Run `python -m unittest -v test_prompt.py` before opening a pull request.

Do not commit model weights, generated media, local archives, logs, PID files, or credentials. Tests must not require a GPU or download models.
