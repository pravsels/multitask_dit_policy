from pathlib import Path


def test_category1_submit_script_defines_expected_variants_and_logging():
    script_path = Path(__file__).resolve().parents[1] / "slurm" / "category1_core_diffusion.sh"
    script_text = script_path.read_text()

    assert 'study_name="coffee_capsules_category1_core_diffusion"' in script_text
    assert 'variants=(' in script_text
    for variant in [
        "baseline",
        "ddpm_20",
        "ddpm_50",
        "ddpm_rope",
        "ddim_20",
        "ddim_50",
        "ddim_rope_20",
        "ddim_rope_50",
    ]:
        assert variant in script_text

    assert "sbatch --parsable" in script_text
    assert 'run_logs/${study_name}' in script_text
    assert '--run_name=${run_name}' in script_text
