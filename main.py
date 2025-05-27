import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from features.detector import Detector
from features.descriptor import Descriptor
from features.matcher import Matcher
from utils.image_io import ImageIO

# Parameters
DATASET_DIR = "dataset"
QUERY_IMG_PATH = "dataset/obj_20/60.png"
MAX_OBJECTS = 20
MAX_VIEWS_PER_OBJECT = 5

image_io = ImageIO()

def match_candidate(candidate_path, img_query, kps_query, desc_query):
    try:
        img_candidate = cv2.imread(candidate_path)
        if img_candidate is None:
            return None

        if len(img_candidate.shape) == 3:
            img_candidate = cv2.cvtColor(img_candidate, cv2.COLOR_BGR2GRAY)

        det = Detector()
        kps, _ = det.detect_keypoints(img_candidate)
        desc = Descriptor(kps, img_candidate)
        desc.compute_descriptors()

        matcher = Matcher(ratio_thresh=0.75)
        matches = matcher.match_descriptors(desc_query.descriptors, desc.descriptors)

        return {
            "path": candidate_path,
            "image": img_candidate,
            "kps": kps,
            "matches": matches,
            "score": len(matches),
            "desc_len": len(desc.descriptors),
            "kp_len": len(kps)
        }
    except Exception as e:
        print(f"[ERROR] {candidate_path}: {e}")
        return None

def main():
    img_query = cv2.imread(QUERY_IMG_PATH)
    if img_query is None:
        print(f"[ERROR] Could not read query image at {QUERY_IMG_PATH}")
        return

    if len(img_query.shape) == 3:
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)

    det_query = Detector()
    kps_query, _ = det_query.detect_keypoints(img_query)
    desc_query = Descriptor(kps_query, img_query)
    desc_query.compute_descriptors()
    candidate_paths = []
    for i, obj_folder in enumerate(sorted(os.listdir(DATASET_DIR))):
        if i >= MAX_OBJECTS:
            break
        obj_path = os.path.join(DATASET_DIR, obj_folder)
        if not os.path.isdir(obj_path):
            continue
        for j, view_file in enumerate(sorted(os.listdir(obj_path))):
            if j >= MAX_VIEWS_PER_OBJECT:
                break
            candidate_paths.append(os.path.join(obj_path, view_file))

    best_result = None
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(match_candidate, path, img_query, kps_query, desc_query) for path in candidate_paths]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is None:
                continue
            print(f"[DONE] {result['path']} → KPs: {result['kp_len']}, Descs: {result['desc_len']}, Matches: {result['score']}")
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result

    if best_result:
        print(f"\n[RESULT] Best match: {best_result['path']} with {best_result['score']} good matches.")
        result_img = image_io.draw_matches(img_query, kps_query, best_result["image"], best_result["kps"], best_result["matches"])
        cv2.imshow("Best Match", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image_io.save_image(result_img)
    else:
        print("\n[RESULT] No good matches found.")

if __name__ == "__main__":
    main()