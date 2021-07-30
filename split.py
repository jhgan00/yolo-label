"""
YOLO 포맷으로 레이블링된 데이터를 train, valid, test 세트로 나누어 저장
"""

import argparse
import glob
import os
import numpy as np
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="imgs", help="이미지와 어노테이션 파일을 포함하는 디렉토리")
    parser.add_argument("--dst_dir", type=str, default="data", help="스플릿된 데이터를 저장할 디렉토리")
    parser.add_argument("--train_size", type=float, default=0.6, help="학습 데이터의 비율")
    parser.add_argument("--valid_size", type=float, default=0.2, help="검증 데이터의 비율")
    parser.add_argument("--copy", type=bool, default=True, help="True 이면 원본 데이터를 복사, False 이면 이동")
    args = parser.parse_args()

    trans = shutil.copy if args.copy else shutil.move

    splits = ['train', 'valid', 'test']
    for split in splits:
        dirname = os.path.join(args.dst_dir, split)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    filenames = [filename for filename in glob.glob(os.path.join(args.src_dir, "*")) if filename.split(".")[-1] != "txt"]

    image_filenames = []
    label_filenames = []

    for filename in filenames:

        label_filename = filename.split(".")[0] + ".txt"
        if os.path.exists(label_filename) and os.path.getsize(label_filename) > 0:
            label_filenames.append(label_filename)
            image_filenames.append(filename)

    label_filenames = np.array(label_filenames)
    image_filenames = np.array(image_filenames)
    num_data = image_filenames.size
    idxs = np.arange(num_data)
    np.random.shuffle(idxs)

    image_filenames = image_filenames[idxs]
    label_filenames = label_filenames[idxs]

    train_idx = int(num_data * args.train_size)
    valid_idx = int(num_data * (args.train_size + args.valid_size))

    trainset = zip(image_filenames[:train_idx], label_filenames[:train_idx])
    validset = zip(image_filenames[train_idx:valid_idx], label_filenames[train_idx:valid_idx])
    testset = zip(image_filenames[valid_idx:], label_filenames[valid_idx:])

    for split, dataset in zip(splits, [trainset, validset, testset]):

        print(split)
        for image_filename, label_filename in dataset:

            base = os.path.split(image_filename)[-1]
            dst = os.path.join(args.dst_dir, split, base)
            trans(image_filename, dst)

            base = os.path.split(label_filename)[-1]
            dst = os.path.join(args.dst_dir, split, base)
            trans(label_filename, dst)
