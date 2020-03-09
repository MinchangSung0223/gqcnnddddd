### 설치

```bash
sudo apt install python3-rtree
apt-get install python3-tk
cd gqcnn
pip3 install trimesh
pip3 install meshrender
pip3 install .
```
### 학습데이터 다운로드
```bash
./scripts/downloads/download_example_data.sh
./scripts/downloads/models/download_models.sh
```
### 테스트(realsense카메라)
```bash
cd ../
python3 realsense_gqcnn.py
```
