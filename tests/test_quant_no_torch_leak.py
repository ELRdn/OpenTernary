"""G18: base/header-only inspect must not import torch via accounting."""

from __future__ import annotations

import subprocess
import sys


def test_accounting_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "import openternary.quant.accounting; "
        "assert 'torch' not in sys.modules, 'torch leaked via accounting'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_inspect_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "import openternary.inspect.engine; "
        "assert 'torch' not in sys.modules, 'torch leaked via inspect.engine'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_quant_init_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "import openternary.quant; "
        "assert 'torch' not in sys.modules, 'torch leaked via quant.__init__'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
