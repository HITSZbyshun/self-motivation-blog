# Debug



在本地端，使用pycharm连接服务器，并进行调试。



原本不会这么复杂，但由于服务器采用slurm管理GPU，因此需要较为复杂的手段。



## 方法一

仅使用单网卡（不在增加一根网线以防增加干扰）

这个方法使用的思路是二段跳

即先从本地用ssh跳到login1（无显卡）

再用相同的方法，将login1作为本地，跳到node85



### 登录yuanxin@login1

- pycharm——文件——远程开发
- 连接到ssh，输入用户名、主机、端口 以及 密码或秘钥



### 连接node85，即使用显卡

- 在新出现的ssh-pycharm中，选择
  - 文件——设置——项目：XX——python解释器
  - 添加解释器——于ssh——新建/现有
    - 主机node85
    - 用户名：yuanxin
    - 端口默认22
    - 选择输入密码



现在使用的ssh解释器，就是带有申请了GPU的python解释器





## 方法二

双网卡法

类似于方法一的思路

因为pycharm的ssh配置不直接支持BindAddress

所以还是把服务器ip中转到local，让pycharm ssh local:localport



### 登录yuanxin@login1

- 在`~\.ssh\config`配置

  - ```bash
    Host 101-3090
        HostName 10.12.218.211
        User yuanxin
        BindAddress 10.192.198.188
        Port 23422
        LocalForward 2222 10.12.218.211:23422

- 现在本地命令行使用`ssh 101-3090`

- pycharm——文件——远程开发
- 连接到ssh，输入用户名、主机、端口 以及 密码或秘钥



### 连接node85，即使用显卡

- 在新出现的ssh-pycharm中，选择
  - 文件——设置——项目：XX——python解释器
  - 添加解释器——于ssh——新建/现有
    - 主机node85
    - 用户名：yuanxin
    - 端口默认22
    - 选择输入密码



现在使用的ssh解释器，就是带有申请了GPU的python解释器

