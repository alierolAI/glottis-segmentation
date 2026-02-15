import os
import json
import pathlib
import subprocess
from pathlib import Path


def set_nnunet_env(base_dir: str):
    """Kaggle/Local fark etmez. nnU-Net klasörlerini set eder."""
    base = Path(base_dir)
    os.environ["nnUNet_raw"] = str(base / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(base / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(base / "nnUNet_results")

    Path(os.environ["nnUNet_raw"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["nnUNet_preprocessed"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["nnUNet_results"]).mkdir(parents=True, exist_ok=True)

    print("nnUNet_raw        :", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results     :", os.environ["nnUNet_results"])


def fix_dataset_json(dataset_json_path: str, file_ending: str = ".nii"):
    """Notebook’taki kritik fix: dataset.json -> file_ending"""
    ds = Path(dataset_json_path)
    data = json.loads(ds.read_text(encoding="utf-8"))
    data["file_ending"] = file_ending
    ds.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("OK -> file_ending set to", data["file_ending"])


def run_cmd(cmd: list[str]):
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Kaggle için: base_dir="/kaggle/working"
    # Local için:  base_dir="./work" (istersen)
    base_dir = os.environ.get("NNUNET_BASE", "/kaggle/working")
    set_nnunet_env(base_dir)

    # Senin dataset path’in (raw içine kopyaladığın Dataset102...)
    # Kaggle notebook’unda: /kaggle/working/nnUNet_raw/Dataset102_BAGLS2D_6K/dataset.json
    dataset_json = os.environ.get(
        "DATASET_JSON",
        f"{os.environ['nnUNet_raw']}/Dataset102_BAGLS2D_6K/dataset.json"
    )
    fix_dataset_json(dataset_json, ".nii")

    # preprocess + train
    run_cmd(["nnUNetv2_plan_and_preprocess", "-d", "102", "--verify_dataset_integrity"])
    run_cmd(["nnUNetv2_train", "102", "2d", "0", "--npz"])


if __name__ == "__main__":
    main()
