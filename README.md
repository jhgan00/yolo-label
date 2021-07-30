# yolo-label

[Yolo_Label](https://github.com/developer0hye/Yolo_Label) 을 이용해 어노테이션된 데이터 조작

- `split.py` : 학습, 검증, 테스트 세트 분리
  
```bash
$ python split.py --src_dir imgs --dst_dir data
```

- `crop.py` : 바운딩 박스 영역을 크롭해서 새로운 파일로 저장
  
```bash
$ python crop.py --src_dir data/train --dst_dir cropped_data/train --label_file labels.txt
```

- `yolo_to_tfrecord.py` : 데이터를 TensorFlow Object Detection API 포맷의 `tfrecord` 파일로 변환 

  
```bash
$ python yolo_to_tfrecord.py --src_dir data/train --dst_dir tfrecord --output_filebase train --label_file labels.txt
```
