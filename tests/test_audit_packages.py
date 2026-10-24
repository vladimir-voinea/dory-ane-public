"""The audit must fail on a package it cannot read.

audit_dory_packages used to swallow every exception from coremltools and report
[] -- "this package will not load" came out as "this package declares no
functions", which is a different (and much less alarming) claim.  A green audit
must mean every package opened and its contents were checked.
"""

import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.audit_dory_packages as audit  # noqa: E402


def _bogus_package(tmp_path, name="dory_recurrent_lut6_infer.mlpackage"):
    """A directory that looks like a package but is not one."""
    package = tmp_path / name
    package.mkdir()
    (package / "Manifest.json").write_text("{not a real manifest")
    return package


def test_an_unreadable_package_raises_instead_of_declaring_nothing(tmp_path):
    package = _bogus_package(tmp_path)
    with pytest.raises(audit.PackageError) as error:
        audit.spec_functions(package)
    message = str(error.value)
    assert package.name in message
    assert "will not load" in message
    # the old behaviour, which must never come back
    assert message != ""


def test_metadata_functions_raises_the_same_way(tmp_path):
    package = _bogus_package(tmp_path, "dory_lm_head_lut6_infer.mlpackage")
    with pytest.raises(audit.PackageError) as error:
        audit.metadata_functions(package)
    assert "dory_lm_head_lut6_infer.mlpackage" in str(error.value)


def test_a_zip_that_is_not_a_model_also_raises(tmp_path):
    # the other shape a broken package takes: a zip without a valid spec
    package = tmp_path / "dory_input_lut6_prefill.mlpackage"
    package.mkdir()
    with zipfile.ZipFile(package / "weights.bin", "w") as archive:
        archive.writestr("nothing.txt", "nothing")
    with pytest.raises(audit.PackageError):
        audit.spec_functions(package)


def test_the_error_is_a_runtime_error_so_main_can_report_it(tmp_path):
    package = _bogus_package(tmp_path)
    try:
        audit.spec_functions(package)
    except audit.PackageError as error:
        assert isinstance(error, RuntimeError)
        assert package.name in str(error)
    else:                                                # pragma: no cover
        raise AssertionError("a broken package must not be accepted")
