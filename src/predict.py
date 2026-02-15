import os
import subprocess


def run_cmd(cmd: list[str]):
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Örnek: test50 prediction
    # input ve output’u env ile değiştirilebilir yaptım
    input_dir = os.environ.get(
        "PRED_INPUT",
        "/kaggle/working/Test50_fixed3d/imagesTs"
    )
    output_dir = os.environ.get(
        "PRED_OUTPUT",
        "/kaggle/working/preds_test50"
    )
    checkpoint = os.environ.get("CHECKPOINT", "checkpoint_best.pth")

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", "102",
        "-c", "2d",
        "-f", "0",
        "-chk", checkpoint
    ]
    run_cmd(cmd)


if __name__ == "__main__":
    main()
