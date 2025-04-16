# datasets



## load_dataset

``` python
from datasets import load_dataset
```



- 数据集存储在各种位置，比如 
  - Hub 
  - 本地计算机的磁盘上
  - Github 存储库中
  - 内存中的数据结构（如 Python 词典和 Pandas DataFrames）中
- 无论您的数据集存储在何处，🤗 Datasets 都为您提供了一种加载和使用它进行训练的方法。



### hugging face hub

使用`datasets.load_dataset()`加载Hub上的数据集。



```python
from datasets import load_dataset
data_files = {"train": "train.csv", "test": "test.csv"}

dataset = load_dataset('lhoestq/demo1',
                       revision="main",
                       data_files=data_files,
                       data_files='en/c4-train.0000*-of-01024.json.gz')
```

- 参数是存储库命名空间和数据集名称（epository mespace and dataset name）
- 根据revision加载指定版本数据集：（某些数据集可能有Git 标签、branches or commits多个版本）
- 使用该data_files参数将数据文件映射到拆分，例如train,validation和test：(如果数据集没有数据集加载脚本，则默认情况下，所有数据都将在train拆分中加载。)
- 使用data_files参数加载文件的特定子集



### 本地和远程文件

本地或远程的数据集，存储类型为csv，json，txt或parquet文件都可以加载



#### CSV

```python
#多个 CSV 文件：
dataset = load_dataset('csv', data_files=['my_file_1.csv', 'my_file_2.csv', 'my_file_3.csv'])

#将训练和测试拆分映射到特定的 CSV 文件：
dataset = load_dataset('csv', data_files={'train': ['my_train_file_1.csv', 'my_train_file_2.csv'] 'test': 'my_test_file.csv'})

#要通过 HTTP 加载远程 CSV 文件，您可以传递 URL：
base_url = "https://huggingface.co/datasets/lhoestq/demo1/resolve/main/data/"

dataset = load_dataset('csv', data_files={'train': base_url + 'train.csv', 'test': base_url + 'test.csv'})
```



#### JSON

```python
from datasets import load_dataset
dataset = load_dataset('json', data_files='my_file.json')
```





#### 导出

| 文件类型             | 导出方式                                                     |
| -------------------- | ------------------------------------------------------------ |
| CSV                  | datasets.Dataset.to_csv()                                    |
| json                 | datasets.Dataset.to_json()                                   |
| Parquet              | datasets.Dataset.to_parquet()                                |
| 内存中的 Python 对象 | datasets.Dataset.to_pandas() 或者 datasets.Dataset.to_dict() |




## load_from_disk

保存和加载dataset

``` python
encoded_dataset.save_to_disk("path/of/my/dataset/directory")

from datasets import load_from_disk
reloaded_encoded_dataset = load_from_disk("path/of/my/dataset/directory")
```







## 服务器无vpn数据集如何获取？

### 方法一

小数据集在本地下载，然后传输到服务器

```python
from datasets import Dataset, load_dataset, load_from_disk
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
dataset.save_to_disk("/media/theo/c8f79183-f83c-4c2e-8065-ce6580d2a20e/datasets/wikitext") # 保存到该目录下
dataset
```

