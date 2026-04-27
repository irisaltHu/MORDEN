#  A Modern ConvNet for Solar Filament Detection

Official PyTorch implementation of **MORDEN** and the post-processing methods.

## 1.Data preparation
The datasets are temporarily available [here](https://box.nju.edu.cn/d/1327c79cdabb49d58961/), please download and place them into the [data](./data) directory.

## 2.Instruction
We used python 3.9 and conda 24.11.3 on a single RTX H100 GPU.

install required packages:
```shell
pip install -r segmentation/requirements.txt
```

if on linux server, PYTHONPATH may need to be set:
```shell
export PYTHONPATH=$PYTHONPATH:{your_path_to_filament_dir}
```

for training, run:
```shell
python segmentation/train.py
```

for inference with DenseCRF, run:
```shell
python segmentation/inference.py
```

for fragment integration, run:
```shell
python segmentation/postprocessing.py
```

## 3.Performance of our model
| Dataset | Test IoU    | Test F1-Score |
|---------|-------------|---------------|
| MHAS    | 0.696±0.005 | 0.819±0.004   |
| AHAS    | 0.812±0.007 | 0.895±0.004   |

## 4.Visualization Results
More visualization results can be found under the [vis](./vis) directory.