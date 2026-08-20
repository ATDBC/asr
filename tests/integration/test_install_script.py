from pathlib import Path


def test_local_runtime_artifacts_are_gitignored():
    ignored = (
        Path(__file__).parents[2] / ".gitignore"
    ).read_text(encoding="utf-8")

    for entry in (
        ".venv/",
        ".venv-asr/",
        ".venv-overlap/",
        "models/",
        ".pytest_cache/",
        "artifacts/release_validation_corpus/",
    ):
        assert entry in ignored


def test_install_script_is_isolated_and_fail_fast():
    script = (
        Path(__file__).parents[2] / "scripts" / "install_linux.sh"
    ).read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "python3.11 -m venv" in script
    assert ".venv-overlap" in script
    assert '-m pip install -e "$PROJECT_ROOT[dev]"' in script
    assert "TORCH_INDEX_URL" in script
    assert "funasr-transcribe doctor" in script
    assert "sudo pip" not in script
    assert "--system-site-packages" not in script


def test_overlap_dependencies_use_torch_2_compatible_pyannote():
    requirements = (
        Path(__file__).parents[2] / "requirements-overlap.txt"
    ).read_text(encoding="utf-8")

    assert "pyannote.audio==3.3.2" in requirements
    assert "pyannote.audio==2.1.1" not in requirements
    assert "librosa==0.10.2.post1" in requirements
    assert "numpy==1.26.4" in requirements
    assert "pyannote.core==5.0.0" in requirements
    assert "pyannote.database==5.1.0" in requirements
    assert "pyannote.metrics==3.2.1" in requirements
    assert "pyannote.pipeline==3.0.1" in requirements
