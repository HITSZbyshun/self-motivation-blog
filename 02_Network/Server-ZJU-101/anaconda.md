

# anaconda


## 环境激活

``` bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/theo/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/theo/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/theo/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/theo/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

```


## 创建与使用环境

- ``` bash
  conda create -n voxposer-env python=3.9
  conda activate voxposer-env
  ```



## 删除环境

### 方法一：

``` bash
# 第一步：首先退出环境

conda deactivate

# 第二步：查看虚拟环境列表，此时出现列表的同时还会显示其所在路径

conda env list

# 第三步：删除环境

conda env remove -p 要删除的虚拟环境路径
conda env remove -p /home/theo/anaconda3/envs/tfpy36   #我的例子
```

### 方法二：

``` bash
# 第一步：首先退出环境

conda deactivate

# 第二步：删除环境

conda remove -n  需要删除的环境名 --all
```



## 打包

注：anaconda 和 miniconda 可以互相迁移

### 拷贝端

- 进入指定的conda环境
- 下载conda pack
  - `pip install conda-pack`
- 打包需拷贝的环境
  - `conda pack -n <env-name>`
  - 这将生成`<env-name>.tar.gz`

### 粘贴端

- 将`<env-name>.tar.gz`传送到粘贴端
- 在`/home/<user-name>/anaconda or miniconda/envs` 下创建环境文件夹
  - `mkdir <env-name>`
- 解压环境压缩包
  - `tar -xzvf <env-name>.tar.gz -C /home/<user-name>/anaconda or miniconda/envs/<env-name>`
- 已经拷贝完毕了，可以查看结果
  - `conda info -e`



## 修改镜像源

``` bash
# 添加清华conda镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes

# 移除默认源（可选）
conda config --remove channels defaults





######### 例如ping不通清华源，就换源

conda config --remove-key channels  # 先清除现有配置
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

```



``` bash
# 创建pip配置目录
mkdir -p ~/.pip

# 编辑pip配置文件
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF


## 再修改换源
vim ~/.pip/pip.conf

[global]
index-url = https://pypi.mirrors.ustc.edu.cn/simple  
trusted-host = pypi.mirrors.ustc.edu.cn  
```





