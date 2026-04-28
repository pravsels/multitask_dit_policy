from pathlib import Path


def test_run_local_script_supports_separate_source_and_shared_dirs():
    script_path = Path(__file__).resolve().parents[1] / "docker" / "run_local.sh"
    script_text = script_path.read_text()

    assert 'SOURCE_PROJECT_DIR="${MTDP_PROJECT_DIR:-$PROJECT_DIR}"' in script_text
    assert 'SHARED_PROJECT_DIR="${MTDP_SHARED_DIR:-$SOURCE_PROJECT_DIR}"' in script_text
    assert '-v "$SOURCE_PROJECT_DIR/src:/workspace/src"' in script_text
    assert '-v "$SOURCE_PROJECT_DIR/tests:/workspace/tests"' in script_text
    assert '-v "$SHARED_PROJECT_DIR/weights:/workspace/weights"' in script_text
    assert '-v "$SHARED_PROJECT_DIR/data:/workspace/data"' in script_text
    assert '-v "$SHARED_PROJECT_DIR/outputs:/workspace/outputs"' in script_text
