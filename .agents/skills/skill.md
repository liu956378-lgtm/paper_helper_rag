---
name: python-testing
description: Run and diagnose pytest tests for Python projects.
---

# Python Testing

When modifying Python code:

1. Inspect existing tests.
2. Run the relevant pytest tests.
3. If tests fail, inspect the traceback.
4. Identify whether the failure is caused by:
   - implementation
   - test
   - environment
   - dependency
   - data
5. Make the minimal required fix.
6. Run the tests again.
7. Report the final result.

Prefer:

```bash
python -m pytest
