from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path("C:/数据/data_original_box")
CLASSES = ["nobirth", "start", "birth"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_number(path):
    stem = path.stem
    if "_" not in stem:
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def list_frames(folder):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])


def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    print(f"ROOT: {ROOT}")
    print(f"ROOT exists: {ROOT.exists()}")
    all_issues = []

    for split in ["train", "val"]:
        print(f"\n================ {split} ================")
        for cls in CLASSES:
            class_dir = ROOT / split / cls
            print(f"\n[{split}/{cls}] path={class_dir} exists={class_dir.exists()}")
            if not class_dir.exists():
                all_issues.append((split, cls, "missing_class_dir", str(class_dir)))
                continue

            sample_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
            counts = []
            le450 = []
            eq450 = []
            gaps = []
            bad_images = []
            nonstandard_names = []
            first_last = []

            for sample_dir in sample_dirs:
                frames = list_frames(sample_dir)
                counts.append(len(frames))

                if len(frames) <= 450:
                    le450.append((sample_dir.name, len(frames)))
                if len(frames) == 450:
                    eq450.append(sample_dir.name)

                nums = [frame_number(p) for p in frames]
                if any(n is None for n in nums):
                    nonstandard_names.append(sample_dir.name)
                nums = [n for n in nums if n is not None]
                if nums:
                    nums_sorted = sorted(nums)
                    first_last.append((sample_dir.name, len(frames), nums_sorted[0], nums_sorted[-1]))
                    expected = list(range(nums_sorted[0], nums_sorted[-1] + 1))
                    if nums_sorted != expected:
                        gaps.append((sample_dir.name, len(frames), nums_sorted[0], nums_sorted[-1]))

                for img_path in frames[:1] + frames[len(frames) // 2 : len(frames) // 2 + 1] + frames[-1:]:
                    if img_path and not check_image(img_path):
                        bad_images.append(str(img_path))

            if counts:
                counter = Counter(counts)
                print(f"samples: {len(sample_dirs)}")
                print(f"frames min/max/avg: {min(counts)} / {max(counts)} / {sum(counts) / len(counts):.1f}")
                print(f"most common frame counts: {counter.most_common(12)}")
                print(f"<=450 samples: {len(le450)}")
                if le450:
                    print(f"<=450 examples: {le450[:30]}")
                print(f"==450 samples: {len(eq450)}")
                if eq450:
                    print(f"==450 examples: {eq450[:30]}")
                print(f"filename gaps: {len(gaps)}")
                if gaps:
                    print(f"gap examples: {gaps[:20]}")
                print(f"nonstandard names: {len(nonstandard_names)}")
                if nonstandard_names:
                    print(f"nonstandard examples: {nonstandard_names[:20]}")
                print(f"bad checked images: {len(bad_images)}")
                if bad_images:
                    print(f"bad image examples: {bad_images[:20]}")
                print(f"first/last examples: {first_last[:8]}")
            else:
                print("samples: 0")
                all_issues.append((split, cls, "empty_class", str(class_dir)))

            for name, count in le450:
                all_issues.append((split, cls, "frames_le_450", name, count))
            for item in gaps:
                all_issues.append((split, cls, "filename_gap", *item))
            for item in bad_images:
                all_issues.append((split, cls, "bad_image", item))

    print("\n================ SUMMARY ================")
    print(f"total issues: {len(all_issues)}")
    for issue in all_issues[:80]:
        print(issue)


if __name__ == "__main__":
    main()
