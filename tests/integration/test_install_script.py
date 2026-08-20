from pathlib import Path
import tomllib


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
    assert 'PYTHON_BIN="$(printenv PYTHON_BIN || true)"' in script
    assert 'PYTHON_BIN="$(command -v python3.11 || true)"' in script
    assert '"$PYTHON_BIN" -m venv "$CORE_VENV"' in script
    assert ".venv-asr" in script
    assert ".venv-overlap" in script
    assert '-m pip install -e "$PROJECT_ROOT[dev]"' in script
    assert (
        '"$ASR_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"'
        in script
    )
    assert (
        '"$OVERLAP_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"'
        in script
    )
    assert "TORCH_INDEX_URL" in script
    assert "funasr-transcribe doctor" in script
    assert "sudo pip" not in script
    assert "--system-site-packages" not in script


def test_core_dependencies_do_not_install_legacy_funasr():
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text("utf-8")
    )

    dependencies = project["project"]["dependencies"]

    assert all(not item.startswith("funasr") for item in dependencies)
    assert "modelscope[framework]==1.36.3" in dependencies
    assert "modelscope==1.36.3" not in dependencies


def test_asr_environment_installs_pinned_source_and_project_metadata():
    script = (
        Path(__file__).parents[2] / "scripts" / "install_linux.sh"
    ).read_text(encoding="utf-8")

    assert (
        'FIRERED_COMMIT="4e7d9aaf4482a47cec1724807026b9b151926eb5"'
        in script
    )
    assert 'FIRERED_REPOSITORY="$(printenv FIRERED_REPOSITORY || true)"' in script
    assert 'FIRERED_REPOSITORY="https://github.com/FireRedTeam/FireRedASR2S.git"' in script
    assert '"$PYTHON_BIN" -m venv "$ASR_VENV"' in script
    assert 'git -C "$FIRERED_SOURCE" rev-parse HEAD' in script
    assert '"$ASR_PYTHON" -m pip install -e "$FIRERED_SOURCE"' in script
    assert '-r "$PROJECT_ROOT/requirements-asr.txt"' in script
    assert (
        '"$ASR_PYTHON" -m pip install --no-deps -e "$PROJECT_ROOT"'
        in script
    )
    assert 'mktemp -d "$FIRERED_ROOT/.source.download-XXXXXX"' in script
    assert 'mv "$SOURCE_TMP" "$FIRERED_SOURCE"' in script
    assert 'rm -rf -- "$FIRERED_SOURCE"' not in script


def test_install_script_verifies_cuda_in_all_three_environments():
    script = (
        Path(__file__).parents[2] / "scripts" / "install_linux.sh"
    ).read_text(encoding="utf-8")

    for python_name in ("CORE_PYTHON", "ASR_PYTHON", "OVERLAP_PYTHON"):
        assert (
            f'"${python_name}" -c' in script
            and "torch.cuda.is_available()" in script
        )


def test_asr_requirements_supply_missing_upstream_runtime_dependency():
    requirements = (
        Path(__file__).parents[2] / "requirements-asr.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert requirements == [
        "modelscope==1.36.3",
        "huggingface-hub==1.4.1",
        "kaldi-native-fbank==1.15",
    ]


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
