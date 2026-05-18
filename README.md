# SVD 矩阵分解的 Pthread 与 OpenMP 多线程优化实现

本项目是南开大学《并行程序设计》课程实验作业，主题为对矩阵奇异值分解（Singular Value Decomposition, SVD）中的 Golub-Kahan 迭代阶段进行多线程优化。

项目在已有 SIMD 优化版本的基础上，分别实现了 **Pthread** 和 **OpenMP** 两种多线程版本。两种实现均以 GKH 迭代过程中划分出的非平凡活动块为并行任务单元，并在保证数值正确性的前提下，对不同线程数和不同调度策略下的性能进行测试与分析。

---

## 项目结构

```text
.
├── pthread
│   ├── .gitkeep
│   ├── bidiagonalization.cpp
│   ├── bidiagonalization.h
│   ├── givens.h
│   ├── gkh.cpp
│   ├── gkh.h
│   ├── main.cpp
│   ├── matrix.h
│   ├── qsub.sh
│   └── test.sh
│
├── OpenMP
│   ├── .gitkeep
│   ├── bidiagonalization.cpp
│   ├── bidiagonalization.h
│   ├── givens.h
│   ├── gkh.cpp
│   ├── gkh.h
│   ├── main.cpp
│   ├── matrix.h
│   ├── qsub.sh
│   └── test.sh
│
├── .gitignore
└── README.md
```

## 目录说明

### `pthread/`

该目录保存 **Pthread 多线程版本**代码。

主要特点：

- 使用 `<pthread.h>` 实现线程创建与同步；
- 通过 `pthread_create` 创建工作线程；
- 通过 `pthread_join` 等待子线程结束；
- 使用原子任务编号动态分发 GKH 活动块任务；
- 主线程也参与计算，避免只负责等待；
- 线程数通过源码中的 `get_gkh_thread_count()` 控制。

### `OpenMP/`

该目录保存 **OpenMP 多线程版本**代码。

主要特点：

- 使用 OpenMP 编译制导语句实现并行循环；
- 通过 `#pragma omp parallel for` 并行处理 GKH 活动块；
- 使用 `omp_set_num_threads()` 设置线程数；
- 支持不同 OpenMP 调度策略：
  - `schedule(static)`
  - `schedule(dynamic,1)`
  - `schedule(dynamic,2)`
  - `schedule(guided)`
- 调度策略通过源码中的 `OMP_SCHEDULE_KIND` 宏控制。

## 文件说明

两个目录中的文件结构基本一致，主要文件作用如下：

| 文件                    | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `main.cpp`              | 测试主程序，负责构造测试矩阵、调用 SVD 分解流程并输出正确性与耗时结果 |
| `matrix.h`              | 基础矩阵类，实现矩阵存储、访问、随机矩阵生成和矩阵乘法等功能 |
| `bidiagonalization.cpp` | Householder 上二对角化阶段实现                               |
| `bidiagonalization.h`   | 上二对角化阶段接口声明                                       |
| `gkh.cpp`               | Golub-Kahan SVD 迭代核心实现，也是本次多线程优化的主要修改文件 |
| `gkh.h`                 | GKH 迭代接口声明                                             |
| `givens.h`              | Givens 旋转相关辅助函数                                      |
| `test.sh`               | 课程提供的测试脚本                                           |
| `qsub.sh`               | 课程提供的任务提交脚本                                       |
| `.gitkeep`              | 用于保留目录结构                                             |

## 实验环境

正式测试环境为课程提供的 OpenEuler 服务器平台。

| 项目         | 配置                             |
| ------------ | -------------------------------- |
| 操作系统     | OpenEuler                        |
| 硬件架构     | AArch64 / ARMv8                  |
| 处理器       | 华为鲲鹏 920                     |
| 计算资源     | 每个任务 1 个 CPU，包含 8 个核心 |
| 编译器       | GCC                              |
| 优化等级     | `-O2`                            |
| 线程测试规模 | 1、2、4、8 线程                  |

由于课程平台通过 `test.sh` 和 `qsub.sh` 将任务提交到计算节点执行，实验中不直接运行生成的可执行文件，也不修改课程提供的 `test.sh` 和 `qsub.sh`。

## 运行方式

进入对应版本目录后，使用课程标准测试脚本运行。

### Pthread 版本

```
cd pthread
bash test.sh 1 1 8 -O O2 -s 20260410
```

### OpenMP 版本

```
cd OpenMP
bash test.sh 1 1 8 -O O2 -s 20260410
```

参数含义如下：

| 参数          | 含义                         |
| ------------- | ---------------------------- |
| `1`           | 实验编号                     |
| `1`           | 节点数                       |
| `8`           | 核心数                       |
| `-O O2`       | 使用 `O2` 优化等级           |
| `-s 20260410` | 指定随机种子，保证测试可复现 |

## 线程数设置

由于课程服务器上的 `test.sh` 会通过 `qsub` 提交任务，命令行环境变量不一定能够稳定传递到计算节点。因此，本项目没有依赖 `SVD_THREADS` 或 `OMP_NUM_THREADS` 等环境变量控制线程数，而是通过源码中的函数手动设置。

在 `gkh.cpp` 中修改：

```
static int get_gkh_thread_count()
{
    return 8;
}
```

测试不同线程数时，将返回值分别修改为：

```
return 1;
return 2;
return 4;
return 8;
```

然后重新运行：

```
bash test.sh 1 1 8 -O O2 -s 20260410
```

------

## OpenMP 调度策略设置

OpenMP 版本通过 `OMP_SCHEDULE_KIND` 控制调度策略。

在 `OpenMP/gkh.cpp` 中修改：

```
#ifndef OMP_SCHEDULE_KIND
#define OMP_SCHEDULE_KIND 1
#endif
```

不同取值含义如下：

| 宏值 | OpenMP 调度策略       |
| ---- | --------------------- |
| `0`  | `schedule(static)`    |
| `1`  | `schedule(dynamic,1)` |
| `2`  | `schedule(dynamic,2)` |
| `3`  | `schedule(guided)`    |

例如，若需要测试 `guided` 调度策略，可将其修改为：

```
#define OMP_SCHEDULE_KIND 3
```

然后重新运行测试脚本。

------

## 输出结果说明

程序运行后会依次输出多个测试样例的正确性和耗时信息，主要关注以下指标：

- `converged`
- `relative recon error`
- `||U^T U-I||_F`
- `||V^T V-I||_F`
- `diagonal structure error`
- `descending order error`
- `nonnegative diagonal`
- `time bidiagonalization(ms)`
- `time gkh iteration(ms)`
- `结果: PASS / FAIL`

最后会输出汇总结果，例如：

```
==============================
随机种子基值: 20260410
总上二对角化耗时(ms): ...
总GKH迭代耗时(ms): ...
通过: 5 / 5
```

其中：

- `总上二对角化耗时(ms)`：Householder 上二对角化阶段总耗时；
- `总GKH迭代耗时(ms)`：Golub-Kahan 迭代阶段总耗时；
- `通过: 5 / 5`：表示全部 5 个测试样例均通过正确性验证。

本实验的多线程优化主要发生在 GKH 迭代阶段，因此性能分析主要关注 `总GKH迭代耗时(ms)`。

------

## 实验结果概述

在服务器正式测试中，Pthread 版本在 8 线程下取得了较明显的 GKH 阶段加速；OpenMP 版本能够正确实现相同的 block 级并行，并且不同调度策略对性能存在明显影响，其中 `guided` 调度在 8 线程调度策略对比中表现较好。

Profiling 结果显示，程序主要热点集中在矩阵列更新相关函数，说明后续进一步优化可以重点关注矩阵存储布局、列方向访存局部性和更细粒度的安全并行策略。

------

## 说明

本仓库中的 `test.sh` 和 `qsub.sh` 为课程测试与提交环境相关脚本，实验过程中不对其进行修改。不同线程数与 OpenMP 调度策略的测试通过修改源码中的线程数返回值和调度策略宏完成。
