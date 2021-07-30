"""
YOLO 포맷으로 어노테이션된 데이터에서 객체 크롭, 저장
"""

import glob
import os
from skimage import io
import argparse
import cv2


def pad(img):

    old_size = img.shape[:2]  # old_size is in (height, width) format
    desired_size = old_size[0] if old_size[0] >= old_size[1] else old_size[1]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def crop(img, x, y, w, h):

    dh, dw, _ = img.shape

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0: l = 0
    if r > dw - 1: r = dw - 1
    if t < 0: t = 0
    if b > dh - 1: b = dh - 1

    return img[t:b, l:r, :]


if __name__ == "__main__":

    # 1. 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="data/train", help="소스 이미지 & 라벨이 들어있는 디렉토리")
    parser.add_argument("--dst_dir", type=str, default="cropped_data/train", help="결과를 저장할 디렉토리")
    parser.add_argument("--label_file", type=str, default="labels.txt", help="클래스 라벨이 저장된 파일")
    parser.add_argument("--min_size", type=int, default=100, help="크롭 후 최소 이미지 크기")
    parser.add_argument("--pad", type=bool, default=False, help="True 이면 정사각형 패딩")
    args = parser.parse_args()

    # 2. idx -> labels
    with open(args.label_file, "r") as f:
        label_dict = dict()
        for i, line in enumerate(f):
            label = line.strip()
            label_dict[i] = label

    # 3. 결과 저장 디렉토리 생성
    for label in label_dict.values():
        label_dir = os.path.join(args.dst_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    # 4. 파일 읽어서 자르고 저장
    filenames = glob.glob(os.path.join(args.src_dir, '*.txt'))  # 어노테이션 파일
    for filename in filenames:

        # 어노테이션 파일 읽기
        with open(filename, "r") as f:
            annotations = [line.split() for line in f.readlines()]

        # 만약 어노테이션이 존재하면
        if annotations:

            # 해당하는 이미지 읽기
            img_filename = filename[:-4] + ".jpg"
            img = io.imread(img_filename)

            # 이미지에 매겨진 어노테이션을 돌면서 사진 자르고 -> 패딩하고 -> 저장
            counter = 0
            for ann in annotations:

                label = int(ann[0])
                x, y, w, h = [float(x) for x in ann[1:]]
                cropped = crop(img, x, y, w, h)
                H, W, _ = cropped.shape

                # 이미지 최소 크기 100 X 100
                if H < args.min_size or W < args.min_size:
                    continue

                # 잘린 이미지가 최소 크기를 넘기면 저장
                else:
                    cropped = pad(cropped) if args.pad else cropped
                    cropped_filename = os.path.basename(filename)[:-4] + "_" + str(counter) + ".jpg"
                    save_path = os.path.join(args.dst_dir, label_dict[label], cropped_filename)
                    io.imsave(save_path, cropped)
                    counter += 1