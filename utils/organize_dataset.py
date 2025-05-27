import os
import shutil
import re

SOURCE_DIR = "coil-100"
TARGET_DIR = "dataset"

pattern = re.compile(r"obj(\d+)__?(\d+)\.png")

os.makedirs(TARGET_DIR, exist_ok=True)

for filename in os.listdir(SOURCE_DIR):
    match = pattern.match(filename)
    if match:
        obj_id = int(match.group(1))
        angle = int(match.group(2))

        obj_folder = os.path.join(TARGET_DIR, f"obj_{obj_id}")
        os.makedirs(obj_folder, exist_ok=True)

        new_filename = f"{angle}.png"
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(obj_folder, new_filename)

        shutil.copy(src_path, dst_path)
        print(f"Copied {filename} → obj_{obj_id}/{new_filename}")
    else:
        print(f"Skipped unrecognized file: {filename}")

        