### 설치

```bash
sudo apt install python3-rtree
apt-get install python3-tk
cd gqcnn
pip3 install trimesh
~pip3 install meshrender==0.0.7~
git clone https://github.com/BerkeleyAutomation/meshrender.git

pip3 install .
```
### librealsense install
```bash
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
```
Ubuntu 16 LTS:
```bash
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u
```
Ubuntu 18 LTS:
```bash
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
```
Ubuntu 20 LTS:
```bash
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo focal main" -u

```
```bash
 sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
 pip3 install pyrealsense2
 pip3 install tqdm
 pip3 install torch
 pip3 install pycocotools
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
