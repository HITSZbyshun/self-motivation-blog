# 初始化配置



---



## tmux

在以下内容中，为方便表示：

''，''  代表先后按下
''+''    代表同时按下

prefix=前缀键=`Ctrl+b`=@



### Session/工作空间

- 开一个session
  - tmux
  - tmux -s <session-name>
- 暂退session
  - @，d
- 列出所有session
  - tmux ls
- 进入某session
  - tmux attach -t 0~9
- 重命名session
  - tmux rename-session -t 0~9 <new-name>

- 杀死session
  - tmux kill-session -t 0~9
  - tmux kill-session -t <session-name>




### Window/窗口

- 新建window
  - @ ，c

- 切换window
  - @ ，0~9     ：切换到特定数字window
  - @，p  ：向左切换window
  - @，n  ：向右切换window
- 查看所有window
  - @，w，上下  ：查看每个window的内容

- 关闭window
  - @，&




### Pane/分屏

- 新建pane
  - @，%    ：创建左右pane
  - @，“      ：创建上下pane

- 切换pane
  - @， 上下左右   ： 手动移动聚焦的pane
  - @，q，0~9    ：  跳转聚焦的pane
- 放大/缩小聚焦的pane
  - @，z  ：按一次放大，再按缩小
- 关闭pane
  - @，x
  - 关闭所有pane相当于关闭window



---



## SSH



### 服务器免密登录

- 在本机上生成ssh密钥对
  - `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`
    - `-t rsa` 指定密钥类型为 RSA.
    - `-b 4096` 指定密钥长度为 4096 位，提供更强的安全性.
    - `-C "your_email@example.com"` 为密钥添加一个标签，方便识别.
    - 按提示操作，系统会询问你保存密钥的位置，默认位置是 `~/.ssh/id_rsa`，你可以直接按回车接受默认位置.
    - 如果你不想为密钥设置密码，直接按回车跳过密码设置步骤；否则，输入密码并确认.
- 将密钥复制到服务器
  - `ssh-copy-id -i ~/.ssh/id_rsa.pub user@server_ip -p port`
    - `user` 是你在服务器上的用户名.
    - `server_ip` 是服务器的 IP 地址.
    - `-i ~/.ssh/id_rsa.pub` 指定公钥文件路径，如果你使用了默认路径，可以省略 `-i` 参数.
    - 运行命令后，系统会提示你输入服务器用户的密码，输入密码后，公钥将被添加到服务器的 `~/.ssh/authorized_keys` 文件中.

- 测试免密登录
  - `ssh user@server_ip -p port`



## github免密上传

### 法1：

- 在本机上生成ssh密钥对
  - `ssh-keygen -t ed25519 -C "your_email@example.com"`

- 将公钥添加到Github账户

  - 打开生成的公钥文件（例如 `~/.ssh/id_ed25519.pub`），复制其内容。
  - 登录 GitHub，进入 **Settings** -> **SSH and GPG keys**，点击 **New SSH key**，将公钥内容粘贴到 **Key** 字段中，然后保存。

- 更改远程仓库URL为ssh格式

  - 检查当前远程仓库 URL：
    - `git remote -v`
  - 如果 URL 是 HTTPS 的，将其改为 SSH 格式：
    - `git remote set-url origin git@github.com:username/repo.git`
    - 将`username`和`repo`替换为你的 GitHub 用户名和仓库名称。



### 法2：

- 配置Git凭证存储
  - 运行以下命令配置 Git 凭证存储
    - `git config --global credential.helper store`
    - 这会在用户主目录下的 `.gitconfig` 文件中添加凭证存储配置
  - 第一次推送输入凭证
    - 第一次使用 `git push` 时，系统会提示你输入用户名和密码。输入后，Git 会将凭证存储起来，以后推送时不再需要输入



---



## 代理

### 一机代理

- 设置代理

```bash
# HTTP/HTTPS 代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

# Socks5 代理
export all_proxy=socks5://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7897

# 如果要持久化，可以将上述命令添加到 ~/.bashrc 或 ~/.zshrc

# 或者一步到位
export all_proxy=http://localhost:7897
export all_proxy=https://localhost:7897
export all_proxy=socks5://localhost:7897
```

- 取消代理

``` bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
```

此链接还包含 windows代理、git代理、pip代理、conda代理、docker代理、wget、curl、apt代理等常用工具的代理配置：https://blog.nim.im/documents/%E5%B8%B8%E7%94%A8%E4%BB%A3%E7%A0%81%E5%9D%97/Proxy.html



### 二机代理：服务器与本机

#### 法1：反向代理

注：本机代表用户自己电脑，服务器代表服务器

##### 本机端

clash等软件打开“局域网连接”

![image-20250108152938252](./ConfigInit.assets/image-20250108152938252.png)

##### 服务器端

终端输入

```bash
export all_proxy=http://本机ip:7897
export all_proxy=https://本机ip:7897
export all_proxy=socks5://本机ip:7897
```



#### 法2：正向代理

##### 本机端

``` bash
ssh -R serverport:localhost:clashport servername@serverip -N -p sshport

ssh -R 1080:localhost:7897 yuanxin@10.12.218.211 -N -p 23422
```

- `-R` 是 SSH 的反向端口转发选项。

- `1080:localhost:7897`指定了端口转发的映射关系：

  - `1080`：远程服务器上的端口，表示服务器上的 1080 端口将被转发。
  - `localhost`：本地主机的地址，表示本地主机上的 7897 端口将被映射到远程服务器的 1080 端口。
  - `7897`：本地主机上的端口，表示本地的 7897 端口将被远程服务器的 1080 端口访问。

- 这意味着，当远程服务器上的 1080 端口被访问时，实际上会连接到本地主机的 7897 端口

  

##### 服务器端

```bash
export all_proxy=http://localhost:7897
export all_proxy=https://localhost:7897
export all_proxy=socks5://localhost:7897
```

