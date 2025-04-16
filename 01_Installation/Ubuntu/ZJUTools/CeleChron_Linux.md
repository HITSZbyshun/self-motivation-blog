# ubuntu20 部署 Celechron



Celechron 是 服务于浙大学生的时间管理器

Celechron 工程地址 https://github.com/Celechron/Celechron



## 安装必要依赖

``` bash
sudo apt update
sudo apt install -y git cmake build-essential qtbase5-dev qtdeclarative5-dev qtquickcontrols2-5-dev qml-module-qtquick2 qml-module-qtquick-controls2 qml-module-qtquick-layouts qml-module-qtquick-window2
sudo apt install -y clang cmake ninja-build pkg-config libgtk-3-dev liblzma-dev
sudo apt install -y libgtk-3-dev
```



## Flutter 安装

### 安装FlutterSDK

``` bash
git clone https://github.com/flutter/flutter.git -b stable
echo 'export PATH="$PATH:$HOME/flutter/bin"' >> ~/.zshrc  # 如果你用 zsh（默认 Ubuntu 20.04），添加环境变量
source ~/.zshrc  # 重新加载配置
flutter --version  # 验证安装
flutter upgrade # 确保最新
```



### 启用 Linux 桌面支持

``` bash
flutter config --enable-linux-desktop
flutter doctor
```

确保出现` [✓] Linux toolchain - develop for Linux desktop`



## 部署Celechron

``` bash
git clone https://github.com/Celechron/Celechron.git
```

### Flutter 编译 Linux

```bash
flutter pub get
flutter build linux --release
```

### 测试是否可以直接运行

``` bash
# 直接运行
./build/linux/x64/release/bundle/celechron
```

### 安装到系统

``` bash
sudo cp -r ./build/linux/x64/release/bundle /opt/celechron
sudo ln -s /opt/celechron/celechron /usr/local/bin/celechron
```

### 进一步安装到桌面

``` bash
sudo vim /usr/share/applications/celechron.desktop
```

输入

``` bash
[Desktop Entry]
Name=Celechron
Comment=课程表应用
Exec=/opt/celechron/celechron
Icon=/opt/celechron/data/flutter_assets/assets/logo.png
Terminal=false
Type=Application
Categories=Utility;Education;
StartupWMClass=celechron
```

赋予可执行权限

``` bash
sudo chmod +x /usr/share/applications/celechron.desktop
```

更新应用菜单

``` bash
sudo update-desktop-database
```

