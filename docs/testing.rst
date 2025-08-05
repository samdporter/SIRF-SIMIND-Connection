.. _testing:

Testing
=======

Running Tests
--------------

To run tests locally:

```bash
python -m pytest tests/ -v -m "not integration"
python -m pytest tests/ -v -m "integration"
```

Continuous Integration
-----------------------

GitHub Actions is used to run tests automatically. Refer to `.github/workflows/tests.yml` for details.
