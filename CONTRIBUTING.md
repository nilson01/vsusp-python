# Contributing

Report a defect with a runnable example, input shapes, versions and the complete
error message. For a method change, cite the source and state the intended
statistical behavior. Add a focused test with an independent expected result.

From a checkout, install `python -m pip install ".[test]"` and run `python -m pytest`.
Keep changes small enough to review. Numerical agreement and statistical
performance require different evidence; retain failures and unsupported cases.
