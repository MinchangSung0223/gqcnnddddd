## install & test
```bash
sudo apt install python3-rtree
apt-get install python3-tk
cd gqcnn
pip3 install trimesh
pip3 install meshrender
pip3 install .

./scripts/downloads/download_example_data.sh
./scripts/downloads/models/download_models.sh
cd ../
python3 realsense_gqcnn.py
```
