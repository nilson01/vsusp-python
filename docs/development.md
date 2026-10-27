# Development

Install and run checks from a source checkout:

```sh
python -m pip install ".[test]"
python -m pytest
```

Build distributions with `python -m build`. The package uses a `src/` layout;
installed-wheel checks must run outside the checkout. Optional R comparisons
use an isolated installation of the repaired VsusP development package and do
not add R to runtime dependencies.

Report defects with a small example, input shapes, package versions and the
complete error message. Keep tests focused on behavior and independent
calculations. Method changes need a source and a stated statistical contract.
