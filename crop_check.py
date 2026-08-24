import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# مسیر پوشه دیتاست
DATASET_DIR = Path("dataset")


def parse_annotation(annotation_path: Path):
    """
    استخراج مختصات باکس‌ها و برچسب از فایل XML یا HTML (با ساختار Pascal VOC)
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    except ET.ParseError as error:
        print(f"خطا در خواندن فایل annotation: {annotation_path.name}")
        print(error)
        return []

    boxes = []
    for obj_idx, obj in enumerate(root.findall("object"), start=1):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")

        if bndbox is None:
            continue

        try:
            label = name_elem.text.strip() if name_elem is not None and name_elem.text else "Unknown"
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append({
                "index": obj_idx,
                "label": label,
                "bbox": (xmin, ymin, xmax, ymax)
            })
        except (AttributeError, TypeError, ValueError):
            continue

    return boxes


def find_image_for_annotation(annotation_path: Path):
    """
    پیدا کردن تصویر متناظر با فایل annotation بر اساس نام فایل (stem)
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # ۱. بررسی در همان پوشه‌ای که فایل annotation قرار دارد
    for ext in image_extensions:
        img_path = annotation_path.with_suffix(ext)
        if img_path.exists():
            return img_path

    # ۲. بررسی در تمام زیرپوشه‌های دیتاست
    for img_path in DATASET_DIR.rglob("*"):
        if (
            img_path.is_file()
            and img_path.stem.lower() == annotation_path.stem.lower()
            and img_path.suffix.lower() in image_extensions
        ):
            return img_path

    return None


def get_all_pairs():
    """
    پیدا کردن تمام جفت‌های معتبر (تصویر + annotation)
    """
    annotation_files = sorted(
        [
            p for p in DATASET_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in [".xml", ".html"]
        ],
        key=lambda x: str(x).lower()
    )

    pairs = []
    for ann_path in annotation_files:
        img_path = find_image_for_annotation(ann_path)
        if img_path is not None:
            pairs.append({
                "image_path": img_path,
                "annotation_path": ann_path
            })

    return pairs


def select_pair(pairs):
    """
    نمایش لیست جفت‌ها و دریافت شماره جفت انتخابی از کاربر
    """
    print(f"\nتعداد کل جفت‌های یافت‌شده: {len(pairs)}\n")
    print("-" * 75)
    for idx, pair in enumerate(pairs, start=1):
        print(f"[{idx:3d}] Image: {pair['image_path'].name:<28} | Annotation: {pair['annotation_path'].name}")
    print("-" * 75)

    while True:
        choice = input(f"\nشماره جفت مورد نظر را وارد کنید (1 تا {len(pairs)} - یا q برای خروج): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(pairs):
                return pairs[idx - 1]
        print(f"ورودی نامعتبر! لطفاً عددی بین 1 تا {len(pairs)} وارد کنید.")


def select_box(boxes):
    """
    اگر در تصویر چند گاو/آبجکت باشد، شماره باکس را می‌پرسد
    """
    if len(boxes) == 1:
        return boxes[0]

    print("\nاین تصویر شامل چند باکس (آبجکت) است:")
    for i, b in enumerate(boxes, start=1):
        print(f"  {i}. BCS/Label: {b['label']} | BBox: {b['bbox']}")

    while True:
        choice = input(f"\nشماره باکس مورد نظر را انتخاب کنید (1 تا {len(boxes)} - یا q برای خروج): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(boxes):
                return boxes[idx - 1]
        print(f"ورودی نامعتبر! لطفاً عددی بین 1 تا {len(boxes)} وارد کنید.")


def visualize_pair(pair):
    img_path = pair["image_path"]
    ann_path = pair["annotation_path"]

    # ۱. خواندن تصویر
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"خطا: تصویر {img_path.name} قابل خواندن نیست.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # ۲. خواندن باکس‌ها
    boxes = parse_annotation(ann_path)
    if not boxes:
        print(f"هیچ باکسی در {ann_path.name} یافت نشد.")
        return

    # ۳. انتخاب باکس
    selected_box = select_box(boxes)
    if selected_box is None:
        return

    xmin, ymin, xmax, ymax = selected_box["bbox"]
    label = selected_box["label"]

    # ۴. کنترل خارج نشدن مختصات از ابعاد تصویر (Clipping)
    xmin = max(0, min(xmin, w_img - 1))
    ymin = max(0, min(ymin, h_img - 1))
    xmax = max(1, min(xmax, w_img))
    ymax = max(1, min(ymax, h_img))

    if xmin >= xmax or ymin >= ymax:
        print(f"خطا: ابعاد باکس نامعتبر است: [{xmin}, {ymin}, {xmax}, {ymax}]")
        return

    # ۵. کراپ در حافظه (بدون ذخیره در دیسک)
    cropped_img = img_rgb[ymin:ymax, xmin:xmax]

    # ۶. رسم مستطیل و برچسب روی تصویر اصلی
    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=3)
    cv2.putText(
        img_with_box,
        f"BCS: {label}",
        (xmin, max(25, ymin - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # ۷. نمایش دو پنجره کنار هم
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_with_box)
    axes[0].set_title(f"Original: {img_path.name}\nBBox: [{xmin}, {ymin}, {xmax}, {ymax}]")
    axes[0].axis("off")

    axes[1].imshow(cropped_img)
    axes[1].set_title(f"Cropped (BCS: {label})\nShape: {cropped_img.shape[1]}x{cropped_img.shape[0]} (WxH)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if not DATASET_DIR.exists():
        print(f"پوشه '{DATASET_DIR.resolve()}' پیدا نشد! مطمئن شو در ریشه پروژه قرار داری.")
        return

    pairs = get_all_pairs()
    if not pairs:
        print("هیچ جفت تصویر و XML/HTML پیدا نشد. ساختار پوشه dataset را چک کنید.")
        return

    pair = select_pair(pairs)
    if pair:
        visualize_pair(pair)


if __name__ == "__main__":
    main()
