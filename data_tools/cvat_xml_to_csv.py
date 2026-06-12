import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_cvat_xml_to_csv(xml_path, csv_path, video_name=None, include_outside=False, with_video_name=True):
    xml_path = Path(xml_path)
    csv_path = Path(csv_path)
    if video_name is None:
        video_name = xml_path.stem

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for track in root.findall("track"):
        instance_id = track.get("id", "")
        label = track.get("label", "")

        for box in track.findall("box"):
            outside = box.get("outside", "0")
            if not include_outside and outside == "1":
                continue

            row = [
                int(box.get("frame", "0")),
                instance_id,
                label,
                float(box.get("xtl", "0")),
                float(box.get("ytl", "0")),
                float(box.get("xbr", "0")),
                float(box.get("ybr", "0")),
            ]
            if with_video_name:
                row = [video_name] + row
            rows.append(row)

    if with_video_name:
        rows.sort(key=lambda x: (x[1], str(x[2])))
    else:
        rows.sort(key=lambda x: (x[0], str(x[1])))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if with_video_name:
            writer.writerow(["video_name", "frame_idx", "instance_id", "label", "x1", "y1", "x2", "y2"])
        else:
            writer.writerow(["frame_idx", "instance_id", "label", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert CVAT annotations.xml to flat CSV.")
    parser.add_argument("--xml", default=None, help="Path to one CVAT annotations.xml")
    parser.add_argument("--xml_dir", default=None, help="Folder containing CVAT xml files")
    parser.add_argument("--csv", default=None, help="Output CSV path for one XML")
    parser.add_argument("--csv_dir", default=None, help="Output folder for batch conversion")
    parser.add_argument("--video_name", default=None, help="Video name to write into CSV")
    parser.add_argument("--no_video_name", action="store_true", help="Do not write video_name column")
    parser.add_argument("--include_outside", action="store_true", help="Keep boxes with outside=1")
    args = parser.parse_args()

    with_video_name = not args.no_video_name

    if args.xml_dir:
        xml_dir = Path(args.xml_dir)
        csv_dir = Path(args.csv_dir) if args.csv_dir else xml_dir
        xml_files = sorted(xml_dir.glob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"No XML files found in {xml_dir}")
        total = 0
        for xml_path in xml_files:
            csv_path = csv_dir / f"{xml_path.stem}.csv"
            count = convert_cvat_xml_to_csv(
                xml_path=xml_path,
                csv_path=csv_path,
                video_name=args.video_name,
                include_outside=args.include_outside,
                with_video_name=with_video_name,
            )
            total += count
            print(f"Saved {count} rows to {csv_path}")
        print(f"Done. Converted {len(xml_files)} XML files, {total} rows in total.")
        return

    if not args.xml:
        raise ValueError("Please provide --xml or --xml_dir")

    xml_path = Path(args.xml)
    csv_path = Path(args.csv) if args.csv else xml_path.with_suffix(".csv")
    count = convert_cvat_xml_to_csv(
        xml_path=xml_path,
        csv_path=csv_path,
        video_name=args.video_name,
        include_outside=args.include_outside,
        with_video_name=with_video_name,
    )
    print(f"Saved {count} rows to {csv_path}")


if __name__ == "__main__":
    main()
