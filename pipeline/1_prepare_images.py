import os
import re
from PIL import Image
from glob import glob

# Input and output folders
# TODO: UPDATE BEFORE EVERY RUN!!!
src_dir = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Source_Data/Images"
dst_base = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Pre_Process/Images"

# Mapping from original channel number → MOAProfiler filenames
# TODO: UPDATE BEFORE EVERY RUN!!!
channel_map = {
    "1": "ch1.tif",  # DNA
    "2": "ch2.tif",  # ER
    "3": "ch3.tif",  # RNA
    "4": "ch4.tif",  # AGP
    "5": "ch5.tif",  # Mito
}
# channel_map = {
#     "2": "ch1.tif",  # DNA
#     "3": "ch2.tif",  # ER
#     "4": "ch3.tif",  # RNA
#     "5": "ch4.tif",  # AGP
#     "6": "ch5.tif",  # Mito
# }

# Target crop size
target_size = (1080, 1080)

def parse_filename(fname):
    """Extract well (e.g., B07), site, and channel number from filename."""
    m = re.match(r"r(\d{2})c(\d{2})f(\d{2})p\d{2}-ch(\d+)sk\d+fk\d+fl\d+_corr\.tiff", fname)
    if not m:
        return None
    row_idx, col_str, site_str, ch_str = m.groups()
    row_letter = chr(ord("A") + int(row_idx) - 1)
    site = int(site_str)
    well = f"{row_letter}{col_str}"
    return well, site, ch_str

def center_crop(img: Image.Image, target_size=(1080, 1080)) -> Image.Image:
    """Center crop to (1080, 1080) from any rectangular image."""
    width, height = img.size
    target_w, target_h = target_size
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))

def main():
    # Collect files by well + site
    files_dict = {}  # {(well, site): {ch: filepath}}
    for fname in os.listdir(src_dir):
        if not fname.endswith(".tiff"):
            continue
        parsed = parse_filename(fname)
        if not parsed:
            print(f"⚠️ Skipping unmatched filename: {fname}")
            continue
        well, site, ch = parsed
        key = (well, site)
        files_dict.setdefault(key, {})[ch] = os.path.join(src_dir, fname)

    # Process each well+site group
    for (well, site), ch_files in sorted(files_dict.items()):
        dst_folder = os.path.join(dst_base, f"{well}_s{site}")
        os.makedirs(dst_folder, exist_ok=True)

        for orig_ch, new_fname in channel_map.items():
            if orig_ch not in ch_files:
                print(f"⚠️ Missing channel {orig_ch} for {well} site {site}, skipping")
                continue

            src_path = ch_files[orig_ch]
            dst_path = os.path.join(dst_folder, new_fname)

            try:
                img = Image.open(src_path)
                cropped = center_crop(img, target_size)
                cropped.save(dst_path)
                print(f"✅ Saved cropped {dst_path}")
            except Exception as e:
                print(f"❌ Error processing {src_path}: {e}")

    print("✅ Finished all image processing.")

if __name__ == "__main__":
    main()

