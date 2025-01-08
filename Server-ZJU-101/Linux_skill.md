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



