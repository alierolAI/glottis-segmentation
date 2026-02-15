from pathlib import Path
import numpy as np
import nibabel as nib


def fix_nii_to_3d(src: Path, dst: Path):
    nii = nib.load(str(src))
    arr = np.asarray(nii.get_fdata())

    if arr.ndim == 2:
        arr = arr[..., None]  # (H,W) -> (H,W,1)
    elif arr.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected ndim={arr.ndim} for {src.name}")

    new = nib.Nifti1Image(arr, affine=nii.affine)
    nib.save(new, str(dst))


def main():
    SRC_DS = Path("/kaggle/input/testing/nnUNet_raw/Dataset102_BAGLS2D_TEST50")
    SRC_IMG = SRC_DS / "imagesTs"
    SRC_LBL = SRC_DS / "labelsTs"

    OUT_DS = Path("/kaggle/working/Test50_fixed3d")
    OUT_IMG = OUT_DS / "imagesTs"
    OUT_LBL = OUT_DS / "labelsTs"
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(SRC_IMG.glob("*.nii")) + list(SRC_IMG.glob("*.nii.gz")))
    lbl_files = sorted(list(SRC_LBL.glob("*.nii")) + list(SRC_LBL.glob("*.nii.gz")))

    print("Images:", len(img_files))
    print("Labels:", len(lbl_files))

    for f in img_files:
        fix_nii_to_3d(f, OUT_IMG / f.name)

    for f in lbl_files:
        fix_nii_to_3d(f, OUT_LBL / f.name)

    # quick check
    p = next(OUT_IMG.glob("*.nii*"))
    a = np.asarray(nib.load(str(p)).get_fdata())
    print("Example fixed shape:", a.shape)


if __name__ == "__main__":
    main()
