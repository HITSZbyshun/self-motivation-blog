# Linux_skill

## ls

- `ls -a`：显示所有文件和目录，包括隐藏文件和当前目录（`.`）及父目录（`..`）。
- `ls -A`：显示所有文件和目录，包括隐藏文件，但不包括当前目录（`.`）和父目录（`..`）。



---




## 压缩文件

### gzip

- 压缩文件：`gzip file.txt`

  - 这会生成一个名为`file.txt.gz`的压缩文件，

  - 并删除原始的`file.txt`文件.
  
- 压缩多个文件：`gzip file1.txt file2.txt`

  - 这会生成`file1.txt.gz`和`file2.txt.gz`

  - 并删除原始文件.

### bzip2

   - 压缩文件：`bzip2 file.txt`

     - 这会生成一个名为`file.txt.bz2`的压缩文件
     - 并删除原始的`file.txt`文件.

### zip

   - 压缩文件到一个 zip 文件：`zip archive.zip file1.txt file2.txt`

     - 这会将`file1.txt`和`file2.txt`压缩到`archive.zip`中.

### tar

   - 创建 tar.gz 文件：`tar -czvf archive.tar.gz directory/`

     - 这会将`directory/`目录下的所有内容压缩到`archive.tar.gz`中.

   - 创建 tar.bz2 文件：`tar -cjvf archive.tar.bz2 directory/`

     - 这会将`directory/`目录下的所有内容压缩到`archive.tar.bz2`中.

## 解压缩文件

### gunzip

   - 解压缩 gzip 文件：`gunzip file.txt.gz`

     - 这会解压缩`file.txt.gz`
     
     - 并生成`file.txt`.

### bunzip2

   - 解压缩 bzip2 文件：`bunzip2 file.txt.bz2`

     - 这会解压缩`file.txt.bz2`
     - 并生成`file.txt`.

### unzip

   - 解压缩 zip 文件：`unzip archive.zip`

     - 这会解压缩`archive.zip`到当前目录.

### tar

   - 解压缩 tar.gz 文件：`tar -xzvf archive.tar.gz`

     - 这会解压缩`archive.tar.gz`到当前目录.

   - 解压缩 tar.bz2 文件：`tar -xjvf archive.tar.bz2`

     - 这会解压缩`archive.tar.bz2`到当前目录.



---



## 卸载

- 搜索卸载软件名称
  - `sudo dpkg --list | grep <some_key_words>`
- 卸载程序和配置文件
  - `sudo apt --purge remove <programname>`
- 仅卸载程序
  - `sudo apt remove <programname>`



---

## 进度条

- 复制较大文件时，需要提供进度，来查看复制多少了
  - `rsync -av --progress 源文件夹路径/ 目标文件夹路径/`
    - `-a`：归档模式，表示以递归方式复制文件，并保留文件的权限、所有者、时间戳等属性。
    - `-v`：详细模式，显示详细的信息。
    - `--progress`：显示进度条。





---



## 关进程

1. 在Ubuntu系统中，如果PyCharm卡死，可以通过以下方法手动关闭PyCharm进程：

   ### 方法一：通过终端命令关闭

   1. 打开终端（可以通过快捷键`Ctrl + Alt + T`）。

   2. 输入以下命令来查找PyCharm的进程ID（PID）：

      bash复制

      ```bash
      ps -ef | grep pycharm
      ```

      这个命令会列出所有与PyCharm相关的进程信息，其中第一列是用户，第二列是进程ID（PID）。

   3. 找到PyCharm的PID后，使用`kill`命令终止该进程：

      bash复制

      ```bash
      kill -9 <PID>
      ```

      将`<PID>`替换为实际的进程ID。`-9`参数表示强制终止进程。

   ### 方法二：使用系统监视器关闭

   1. 在桌面右上角的菜单栏中，点击“系统监视器”图标（如果没有，可以通过`Ctrl + Alt + Delete`快捷键，选择“系统监视器”）。
   2. 在“进程”标签页中，找到名为“pycharm”或类似名称的进程。
   3. 选中该进程，然后点击“结束进程”按钮。

   ### 方法三：使用`htop`工具（如果已安装）

   1. 打开终端。

   2. 输入以下命令启动`htop`：

      bash复制

      ```bash
      htop
      ```

   3. 在`htop`界面中，找到与PyCharm相关的进程（通常名称为`pycharm`或`java`）。

   4. 使用方向键选中该进程，然后按`F9`键选择“Kill process”来强制终止进程。

   ### 方法四：通过图形界面的“活动”菜单

   1. 点击Ubuntu桌面左上角的“活动”按钮。
   2. 在搜索栏中输入“系统监视器”并打开它。
   3. 在“进程”标签页中找到PyCharm进程，右键选择“结束进程”。

   如果PyCharm卡死是因为文件损坏或配置问题，建议在下次启动时检查PyCharm的配置文件或尝试重新安装。
