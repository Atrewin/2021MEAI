# 2021MEAI
2021年清深-智能计算夏令营
# 声明版权，请勿转载



## 1. 数据
* 1.1 环境
> pip install requirement.txt
>> requirement.txt 作者整个环境的导出版本，全部安装耗时会很长， 建议直接运行，保存再对应安装
>>> 主要  
>torch==1.7.1
>opencv-python==4.4.0.44
>matplotlib==3.3.2


* 1.2 数据
>> 本项目的图片读取规范和 `torchvision.datasets.ImageFolder` 使用规范保持一致
>> 请将所用图片放入同一个文件夹中，且相同类别在放入相同的子文件夹
>> 如果有新的图片类别出现，请在 `data/data_loder.DataConfig` 中重新配置label和index的关系


## Training

>> python main.py --source source_data_root --target_data_ root --batch_size 16
>>> 如数据集1迁移数据2：python main.py --source data/datasets/train_val/train --data/datasets/NEU-CLS --batch_size 16

## Testing
>> python main.py --source source_data_root --target_data_ root --batch_size 16 --only_test_target --load_ckpt ckpt_path
>>> 如：python main.py --source data/datasets/train_val/train --target data/datasets/NEU-CLS --batch_size 16 --only_test_target --load_ckpt output/9-6-model.bin 


>> 注意path_root可以为绝对路径也可也是相对于项目根目录的相对路径