"""
YOLO 포맷으로 레이블링된 데이터를 TensorFlow Object Detection API 포맷으로 변환
"""

import tensorflow as tf
import glob
import argparse
import os
from skimage import io
import contextlib2
import pathlib
from tqdm import tqdm
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


def convert_coords(x, y, w, h, width, height):

    l = int((x - w / 2) * width)
    r = int((x + w / 2) * width)
    t = int((y - h / 2) * height)
    b = int((y + h / 2) * height)

    l = max(l, 0)
    r = width - 1 if r > width - 1 else r
    t = max(t, 0)
    b = height - 1 if b > height - 1 else b

    return t, l, b, r  # ymin, xmin, ymax, ymin


def create_tf_example(label_path, img_path):

    base, ext = img_path.split(".")
    img = io.imread(img_path)
    height, width, _ = img.shape
    image_format = ext.encode("utf-8")

    with open(label_path, "r") as f:
        classes, classes_text, xmins, xmaxs, ymins, ymaxs = [], [], [], [], [], []

        annotations = [line.split() for line in f.readlines()]

        if not annotations:
            return None

        for ann in annotations:
            label = int(ann[0])
            x, y, w, h = [float(x) for x in ann[1:]]
            ymin, xmin, ymax, xmax = convert_coords(x, y, w, h, width, height)

            classes.append(label)
            classes_text.append(LABEL_DICT[label].encode('utf-8'))
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(img_path.encode('utf-8')),
                'image/source_id': dataset_util.bytes_feature(img_path.encode('utf-8')),
                'image/encoded': dataset_util.bytes_feature(tf.io.gfile.GFile(img_path, 'rb').read()),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes)
            }
        )
    )

    return tf_example


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="data/train", help="학습 이미지와 어노테이션이 들어있는 디렉토리")
    parser.add_argument("--dst_dir", type=str, default="tfrecord", help="tfrecord 데이터를 저장할 디렉토리")
    parser.add_argument("--output_filebase", type=str, default="train", help="tfrecord 파일명")
    parser.add_argument("--label_file", type=str, default="labels.txt", help="YOLO 클래스 이름 파일")
    parser.add_argument("--num_shards", type=int, default=1, help="tfrecord 파일 생성 갯수")
    args = parser.parse_args()

    with open(args.label_file, "r") as f:
        LABEL_DICT = dict()
        for i, line in enumerate(f):
            label = line.strip()
            LABEL_DICT[i] = label

    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)

    with contextlib2.ExitStack() as tf_record_close_stack:

        output_filebase = os.path.join(args.dst_dir, args.output_filebase)
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, args.num_shards
        )

        img_filenames = [filename for filename in glob.glob(os.path.join(args.src_dir, "*")) \
                         if pathlib.Path(filename).suffix in {".jpg", ".jpeg", ".png"}]
        label_filenames = [filename.split(".")[0] + ".txt" for filename in img_filenames]
        total = len(img_filenames)

        for index, (label_path, img_path) in tqdm(enumerate(zip(label_filenames, img_filenames)), total=total):
            tf_example = create_tf_example(label_path, img_path)
            output_shard_index = index % args.num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
