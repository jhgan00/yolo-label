# yolo-label

[Yolo_Label](https://github.com/developer0hye/Yolo_Label) 을 이용해 어노테이션된 데이터 조작

- `split.py` : 학습, 검증, 테스트 세트 분리
- `crop.py` : 바운딩 박스 영역을 크롭해서 새로운 파일로 저장
- `yolo_to_tfrecord.py` : 데이터를 TensorFlow Object Detection API 포맷의 `tfrecord` 파일로 변환 