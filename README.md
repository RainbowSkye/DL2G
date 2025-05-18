# DL2G

## Requisites

* Python =3.7, PyTorch = 1.7.0
* scikit-image = 0.15.0, Pillow = 6.2.2


## Quick Start

### dataset

* 请确保你的数据集文件如下：
  ```
  dataset
   |-train
     |-gt
     |-specular
   |-val
     |-gt
     |-specular
    ```

### train

* 可以通过运行如下代码或者执行**train.sh**脚本来开始训练（`--name`为本次训练自定义的模型名）: 

  ```
  python sr.py -p train -c config/DG2_train.json -gpu 1
  ```


#### test


* 可以通过运行如下代码或者执行**test.sh**脚本来开始训练（`--name`为训练保存的模型名）: 

  ```
  python sr.py -p val -c config/DG2_val.json -gpu 1
  ```

