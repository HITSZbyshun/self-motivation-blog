# SSH



## 服务器免密登录

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



# github免密上传

## 法1：

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



## 法2：

- 配置Git凭证存储
  - 运行以下命令配置 Git 凭证存储
    - `git config --global credential.helper store`
    - 这会在用户主目录下的 `.gitconfig` 文件中添加凭证存储配置
  - 第一次推送输入凭证
    - 第一次使用 `git push` 时，系统会提示你输入用户名和密码。输入后，Git 会将凭证存储起来，以后推送时不再需要输入





# 双网卡登录ssh服务器

- 本人ubuntu20，同时连接了两种网络
  - 有一根网线走高速网络，但不可以访问服务器
  - 有一张无线网卡走校园局域网，可以访问服务器

- 诉求是，平时默认使用网线访问网络，并常年开tun模式；而走服务器则使用网卡。



- 首先，在`~/.ssh/config`中配置

  - ```bash
    # 配置项各意思
    Host <服务器别名>
        HostName <服务器ip>
        User <用户名>
        BindAddress <wifi网卡ip>
        Port <服务器端口号>
        LocalForward <本机端口号> <服务器ip>:<服务器端口号>
    ```

  - ```bash
    # 我的配置
    Host 101-3090
        HostName 
        User yuanxin
        BindAddress 10.192.198.188
        Port 23422
        LocalForward 2222 10.12.218.211:23422
    ```

- 在终端中输入`ssh 101-3090`即可访问服务器

- 如需使用filezilla这样不可以配置ssh -b命令的软件，可以使其走127.0.0.1:2222 yuanxin 密码，即可访问服务器 