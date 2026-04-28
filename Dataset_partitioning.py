# 将 VOC2007 格式数据集转换为 YOLO 格式，并按 8:1:1 划分 train / val / test
# 源数据目录: VOCdevkit/VOC2007/JPEGImages 与 VOCdevkit/VOC2007/Annotations
# 输出目录: VOCdevkit/YOLO_Dataset/ 下的 images 与 labels

import os
import random
import shutil
import xml.etree.ElementTree as ET

# 类别列表（顺序对应 YOLO 标签中的类别 id：0,1,2,...），需与 XML 中 <name> 一致
classes = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]

# 划分比例（百分比整数，三者之和应为 100）
TRAIN_RATIO = 80  # 训练集 80%
VAL_RATIO = 10  # 验证集 10%
TEST_RATIO = 10  # 测试集 10%（余数样本会进测试集，保证总数一致）

# 项目根目录 = 本脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# VOC 数据根目录（与 VOC2007 标准结构一致）
VOC_ROOT = os.path.join(BASE_DIR, "VOCdevkit", "VOC2007")
# 转换并划分后的 YOLO 数据集根目录
YOLO_OUT = os.path.join(BASE_DIR, "VOCdevkit", "YOLO_Dataset")

# 随机种子，固定后每次划分结果一致
RANDOM_SEED = 42

# 支持的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clear_hidden_files(path):
    """删除目录下以 ._ 开头的隐藏文件（常见于 macOS 拷贝产生的垃圾文件）。."""
    if not os.path.isdir(path):
        return
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath) and i.startswith("._"):
            os.remove(abspath)


def _voc_xml_to_yolo_lines(xml_path, skip_difficult=True):
    """读取单个 VOC xml，返回 YOLO 格式的标注行列表。."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return []
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    if w <= 0 or h <= 0:
        return []

    lines = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls_name = name_el.text.strip()
        if cls_name not in classes:
            print(f"[警告] 未知类别 '{cls_name}'，已跳过: {os.path.basename(xml_path)}")
            continue

        diff = obj.find("difficult")
        if skip_difficult and diff is not None and diff.text and int(diff.text) == 1:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.find("xmin").text)
        xmax = float(box.find("xmax").text)
        ymin = float(box.find("ymin").text)
        ymax = float(box.find("ymax").text)

        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)

        cid = classes.index(cls_name)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _collect_samples():
    """返回 [(不含后缀的文件名, 图片绝对路径), ...]，要求同名的 xml 存在。."""
    img_dir = os.path.join(VOC_ROOT, "JPEGImages")
    ann_dir = os.path.join(VOC_ROOT, "Annotations")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"未找到 JPEGImages: {img_dir}")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"未找到 Annotations: {ann_dir}")

    samples = []
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        img_path = os.path.join(img_dir, name)
        if not os.path.isfile(img_path):
            continue
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if os.path.isfile(xml_path):
            samples.append((stem, img_path))
        else:
            print(f"[警告] 无对应 XML，跳过图片: {name}")
    return samples


def _split_list(items):
    """按 TRAIN_RATIO / VAL_RATIO / TEST_RATIO 划分（整数切分，余数进测试集）。."""
    n = len(items)
    if n == 0:
        raise ValueError("没有可用的图片+XML 样本。")

    r_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if r_sum != 100:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO 应等于 100")

    shuffled = items[:]
    random.shuffle(shuffled)

    n_train = n * TRAIN_RATIO // 100
    n_val = n * VAL_RATIO // 100
    n_test = n - n_train - n_val
    while n_train < 1 or n_val < 1 or n_test < 1:
        if n < 3:
            raise ValueError(f"样本过少 (n={n})，无法划分 train/val/test。")
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _ensure_dirs():
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(YOLO_OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_OUT, "labels", split), exist_ok=True)


def _write_split(subset, split_name):
    ann_dir = os.path.join(VOC_ROOT, "Annotations")
    for stem, src_img in subset:
        dst_img = os.path.join(YOLO_OUT, "images", split_name, os.path.basename(src_img))
        shutil.copy2(src_img, dst_img)

        xml_path = os.path.join(ann_dir, stem + ".xml")
        lines = _voc_xml_to_yolo_lines(xml_path, skip_difficult=True)
        lb_path = os.path.join(YOLO_OUT, "labels", split_name, stem + ".txt")
        with open(lb_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


def _write_dataset_yaml():
    """生成 Ultralytics 可直接使用的 dataset.yaml。."""
    path_pos = os.path.normpath(YOLO_OUT).replace("\\", "/")
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    content = (
        f"# 由 Dataset_partitioning.py 生成\n"
        f"path: {path_pos}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names_yaml}\n"
    )
    with open(os.path.join(YOLO_OUT, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(content)

    cls_path = os.path.join(YOLO_OUT, "classes.txt")
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")


def main():
    random.seed(RANDOM_SEED)

    clear_hidden_files(VOC_ROOT)
    clear_hidden_files(os.path.join(VOC_ROOT, "JPEGImages"))
    clear_hidden_files(os.path.join(VOC_ROOT, "Annotations"))

    samples = _collect_samples()
    train_s, val_s, test_s = _split_list(samples)

    if os.path.isdir(YOLO_OUT):
        shutil.rmtree(YOLO_OUT)
    _ensure_dirs()

    print(f"共 {len(samples)} 张（含 XML）-> train:{len(train_s)} val:{len(val_s)} test:{len(test_s)}")
    print(f"输出目录: {YOLO_OUT}")

    _write_split(train_s, "train")
    _write_split(val_s, "val")
    _write_split(test_s, "test")
    _write_dataset_yaml()

    print("完成。训练示例: yolo detect train data=VOCdevkit/YOLO_Dataset/dataset.yaml model=yolov8n.pt")


if __name__ == "__main__":
    main()
