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