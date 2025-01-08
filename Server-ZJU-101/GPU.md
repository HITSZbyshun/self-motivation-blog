# GPU



## 借用显卡

- 命令行交互式借用显卡`srun -N 1 --pty --cpus-per-task=4 -t 5:00 -p gpu-3 --gres=gpu:1 /bin/bash -i`

  - `-t 5:00` 代表借用5分钟，`-t 7-00:00:00` 代表借用7天
  - `-p gpu-3` 代表借用3090，`gpu-2` 2080 `gpu-1` 1080
  - `--gres=gpu:1` 代表借用一块GPU



## 显示借用

`squeue`



## 退还借用

- `scancel 10`: 取消`JOBID`为10的任务组中的全部作业
- `scancel 10_[1-2]`: 取消`JOBID`为10的任务组中`TASK_ID`为1和2的作业
- `scancel 10_0 10_2`: 取消`JOBID`为10的任务组中 `TASK_ID`为0和2的作业
