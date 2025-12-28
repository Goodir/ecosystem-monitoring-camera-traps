import zipfile
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "jnle/wildlife_conservation_camera_trap_dataset"
HF_OUT_DIR = Path("data/camera_trap")

WCS_URL = "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip"
WCS_ZIP_PATH = Path("wcs_camera_traps.json.zip")

def unzip(zip_path: Path, extract_to: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def main():
    # HF dataset
    HF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading HF dataset...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(HF_OUT_DIR),
        local_dir_use_symlinks=False,
    )

    # unzip HF archives
    if not (HF_OUT_DIR / "camera_trap_dataset").exists():
        print("Unzipping HF dataset...")
        for zp in HF_OUT_DIR.rglob("*.zip"):
            if zipfile.is_zipfile(zp):
                unzip(zp, HF_OUT_DIR)
                zp.unlink()

    # WCS zip
    if not WCS_ZIP_PATH.exists():
        print("Downloading WCS zip...")
        urllib.request.urlretrieve(WCS_URL, WCS_ZIP_PATH)

    print("Done.")

if __name__ == "__main__":
    main()
