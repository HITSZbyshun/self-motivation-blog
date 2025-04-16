## 建立新仓库

### 初始化本地仓库

```bash
# 进入你的项目目录
cd /path/to/your/project

# 初始化 Git 仓库
git init
```

### 关联远程仓库

```bash
git remote add origin https://github.com/HITSZbyshun/仓库名.git
```

### 添加、提交、推送

```bash
# 添加所有文件
git add .

# 或添加特定文件
git add 文件名

git commit -m "提交描述"

# 首次推送（指定上游分支）
git push -u origin main

# 后续推送（如果已关联分支）
git push
```



### 如果远程仓库已有文件（需先拉取）

如果远程仓库已有文件（如 README 或 LICENSE），先执行：

```bash
git pull origin main --allow-unrelated-histories
```



### 修改远程仓库名称

在某仓库下，点击setting，第一项就是修改名称



## 分支

### 修改本地分支名称

1. **确保当前在 master 分支**

   ```bash
   git checkout master
   ```

2. **重命名本地分支**

   ```bash
   git branch -m master main
   ```

   `-m` 是 `--move` 的缩写，表示移动/重命名分支

3. **推送新分支到远程仓库**

   ```bash
   git push -u origin main
   ```
   
4. **删除远程的 master 分支**

   ```bash
   git push origin --delete master
   ```





##  Git Submodule

```bash
# 在 A 仓库目录下
git submodule add https://github.com/你的用户名/B仓库.git 目标路径
git commit -m "添加 B 仓库作为子模块"
git push
```
