# 湖师院GPU使用方法



## 登录

- easyconnect  https://vpn.zjhu.edu.cn
- http://172.20.137.99/core/auth/login/



## 借用

- sbatch -N 1 -p zjhu -t 7-00:00:00 --cpus-per-task=3 --gres=gpu:4090:1 ./prune.sh 



## 优雅地使用

EasyConnect的弊端太多

这里采用一种使用docker包裹easyconnect的方法

- 默认安装了docker，以及clash （clash-verge-rev 1.7.x 以上的版本）

- 在终端使用

```bash
sudo docker run --rm --device /dev/net/tun --cap-add NET_ADMIN -ti -p 127.0.0.1:1080:1080 -p 127.0.0.1:8888:8888 -e EC_VER=7.6.3 -e CLI_OPTS="-d https://vpn.zjhu.edu.cn -u cj -p password" hagb/docker-easyconnect:cli
```

- clash中，选择 编辑节点

  ```bash
  prepend:
    - name: 'vpn'
      type: 'socks5'
      server: '127.0.0.1'
      port: 1080
  ```

- clash中，选择 编辑规则

  ```bash
  prepend:
    - 'IP-CIDR,172.20.137.99/32,vpn'
  ```



配置完成后，重启网络环境（很重要，重启电脑也行）

- 实现：
  - 外网ip走代理
  - 国内ip走直连
  - 172.20.137.99ip走easyconnect
