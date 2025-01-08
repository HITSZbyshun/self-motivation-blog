# Tmux

在以下内容中，为方便表示：

''，''  代表先后按下
''+''    代表同时按下

prefix=前缀键=`Ctrl+b`=@



## Session/工作空间

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




## Window/窗口

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




## Pane/分屏

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