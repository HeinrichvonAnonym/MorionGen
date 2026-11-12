# 正式批量监控与交接：20261110_batch_main

批次 `20261110_batch_main`，复用 `20261109_batch_rehearsal`；运行器启动时间 **2026-11-10T00:40:17+08:00**（Asia/Shanghai），workers **8**，启动时 runner PID **1671665**。采样器实际 PID **1672558**，PID 文件 `develop_logs/test_runs/20261110_batch_main/monitor_sampler.pid`，启动命令见下一段。启动前资源基线采于 **2026-11-10T00:40:04+08:00**：数据盘 **1112264495104 bytes（1035.877 GiB）**，系统盘 **62720135168 bytes（58.413 GiB）**，内存 available **20861636608 bytes（19.429 GiB）**，`free -g` available **19**。数据设备 `/dev/sda1`，系统设备 `/dev/nvme0n1p3`。以上为实测值，系统盘变化量统一相对此字节基线计算。

正式启动命令（已执行、退出码 0）：

```bash
PYTHONNOUSERSITE=1 python3 python/scripts/batch_pipeline.py start --batch 20261110_batch_main --reuse-batch 20261109_batch_rehearsal --workers 8
```

采样器启动命令（已执行；实际使用 subprocess 的 argv 方式启动，下面为等价 shell 表达）：

```bash
PYTHONNOUSERSITE=1 nohup setsid nice -n 19 /home/heinrich/anaconda3/bin/python3 /home/heinrich/MorionGen/develop_logs/test_runs/20261110_batch_main/monitor_sampler.py --repo-root /home/heinrich/MorionGen --batch 20261110_batch_main --data-root /media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen --baseline-system-bytes 62720135168 --log /home/heinrich/MorionGen/develop_logs/notes/20261110_batch_monitor_samples.log --pid-file /home/heinrich/MorionGen/develop_logs/test_runs/20261110_batch_main/monitor_sampler.pid --interval-seconds 3600 </dev/null >>/home/heinrich/MorionGen/develop_logs/test_runs/20261110_batch_main/logs/monitor_sampler.log 2>&1 &
```

采样器独立会话已核实，nice=19，PID/PGID/SID 均为 1672558。它只执行 status、df/free/nvidia-smi 和本批评估 worker 的 ps RSS 汇总；不 resume、不发信号、不读结果文件或评估日志。采样日志为 `develop_logs/notes/20261110_batch_monitor_samples.log`，与本笔记分开写入。每 60 分钟采样一次（约定，未经推演），runner 退出后仍采样；analysis done 后追加一次采样并自行退出，PID 文件保留作为记录。采样器不能唤醒已关闭的 agent。

## 接管依据和保护范围

- 项目根目录没有 AGENTS.md；以 develop_rules.md、report_rules.md 和 develop_logs/notes/20261109_batch_pipeline_runbook.md 为依据。运行手册是启动、续跑和故障处理的依据，本笔记保留用户已明确的补充和修正。
- 当前 HEAD `062f7ea0995f99055ed8dbe4ff61a98971a8b5e2`，包含 a7d0cdf、688f775、341085a、062f7ea。启动前冻结检查退出码 0，differences=[]。
- 已修改的已跟踪文件仅 `python/scripts/e_v2_moco_refs.py`，不在本批保护范围或生成入口依赖范围内，保持原状。其他会话的未跟踪文件保持原状，不清理、不提交。
- 未跟踪的 `config/hiphi_object_pilot_selection.json`、`config/hiphi_object_pilot_selection_v2.json`、`config/hiphi_object_pilot_selection_v3.json`、`config/hiphi_object_pilot_selection_v4.json` 会被执行身份读取并校验；未发现其他未跟踪 YAML 和环境锁文件被本批直接使用。
- 运行期间按全部 config 不可变管理；代码实际遍历 config/**/*.json（有少量既定报告文件例外，不依赖例外放行）。其他会话修改 config 可能导致生成队列 resume 拒绝；只报告，不还原他人的文件。根目录 plans/ 只读，正式批次的编排计划也不得人工修改。
- 不泄露主机地址或凭据。所有命令带 PYTHONNOUSERSITE=1，Git 使用 /usr/bin/git；后台启动使用 nohup setsid nice -n 19。当前 python 为 `/home/heinrich/anaconda3/bin/python3`；评估使用编排既定的 OpenSim 环境。
- 不打开任何指标数值、evalkit/error 数值字段、analysis 结果；不打开评估日志全文。只允许在报错时用 rg -n "Traceback|Error" 提取对应阶段日志。不对留出清单评估，不打开留出清单或其输出。
- 彩排各阶段 done、runner_alive=false。彩排 analysis 的 done 对应 688f775 修正后那次编排执行；341085a 修正后的结构检查是随后手动重跑，不能把两次结果混为一次。
- 未找到 evalkit 单 worker（包括腰部 OpenSim）的峰值内存证据。workers=8 已由用户选定；resume 即使传 --workers 也不会改变 start 保存的值。8 相比 10 的评估耗时约增加 25% 只是按并发反比推算，实际未验证。约 3.5–4 天的总运行预期为约定，未经推演，不作为强制停止期限。

## 启动验收

全部检查通过：problems=[]；generate_main 新任务 2017，generate_chois 84，彩排 origin=rehearsal 链接 15（13+2）；生成范围 2116=2101 新任务+15 彩排任务，其中留出生成任务 644。另有 frozen_reuse 历史链接 644，不混入这 2116。

新任务预期：MotionLCM 454、OmniControl 908、CLoSD 238、DART 238、ARDY 120、PRIMAL 59、CHOIS 84。正式批次目录为符号链接，解析后等于 data_root/batch_runs/20261110_batch_main，findmnt 显示 /dev/sda1，非系统盘。传给编排的路径继续使用仓库内链接路径。初始 runner_alive=true，generate_main=running。

采样器先在 develop_logs/test_runs/20261110_batch_monitor_setup/ 下用合成数据验证，再复制到本批目录。7 项测试通过，覆盖进程排除、RSS、告警边界、首次 error 不告警、输出过滤、PID 和最终采样退出。验证没有启动生成或评估。

## 在线 agent 的检查和处置

先确认采样器 PID 对应本批 sampler，检查采样日志最新时间，用 `PYTHONNOUSERSITE=1 rg -n '^ALERT ' develop_logs/notes/20261110_batch_monitor_samples.log` 汇总告警。agent 每小时执行 status、df/free，并在本笔记追加时间戳、当前阶段、各方法 success/failed/timeout/resource_gate/pending/running、runner_alive、评估 evaluated/errors、两盘可用空间、free -g available、系统盘相对基线变化。三个评估阶段另记 worker 数、总 RSS、单 worker 最大 RSS（KiB），仅为采样瞬时值，不是整段运行的真实峰值。

采样器只对 runner_alive=false、阶段 blocked、内存 available<3 GiB、系统盘可用空间减少>5 GiB 加 ALERT。第一条 error 只记数。以下告警阈值和 60 分钟周期都是“约定，未经推演”。

| 情况 | 处置 |
|---|---|
| runner 存活且计数正常推进 | 只记录 |
| runner 退出且阶段停在 running | 先确认本批运行器、队列及 worker 无残留，再按手册 resume --batch 20261110_batch_main |
| resume 因计划、身份或冻结变化拒绝 | 报告，不修改或还原文件 |
| Interrupted running job requires process/evidence reconciliation | 报告用户，等待本批专用核验决定；不直接改状态，不运行旧 reconcile 脚本 |
| blocked 或空间不足 | 报告，不删除文件 |
| 同一方法 failed/timeout 连续增加，或一小时新增>5 | 报告；只用允许的 rg 提取报错；存活运行器不动 |
| 评估阶段内存连续两次<3 GiB | 报告，不 kill |
| 任一评估阶段一小时新增 errors>5 | 报告 |
| 任一评估阶段累计 errors>evaluated×2% | 报告；evaluated=0 时比例记为不可计算，其他规则继续适用 |
| 系统盘可用空间比基线少>5 GiB | 用 timeout 120 du -xsh /tmp ~/.cache 等只读定位；不清理；本批写错盘则报告 |
| 一小时计数不变且 runner 存活 | 检查 nvidia-smi、本批进程和 heartbeat，判断是否等共享 GPU 锁，再报告 |
| 其他情况 | 停止自动处置，通过 AskQuestion 询问用户 |

每阶段第一条 error：agent 下次在线仅执行 `PYTHONNOUSERSITE=1 rg -n "Traceback|Error" develop_logs/test_runs/20261110_batch_main/logs/<stage>.log`，记录是否能判断为零星或系统性失败，同时应用累计/增长规则。无匹配则如实记录无法判断并询问，不扩大读取范围。“停下”是停止自动处置并报告，不终止仍在运行的运行器；采样器继续只读记录。

生成 failed 不人工重跑、不换种子、不替换；保留冻结队列已有的 timeout 重试行为。resource_gate 仅由编排进行同任务同种子的一次重试；之后仍被拒的任务交用户决定。不得操作其他会话进程、文件或 GPU 任务；GPU 通过共享锁排队。

## 中断核验只读参考：不实现、不运行

旧 reconcile_native_interruption.py 写死四个旧批次，不接受本批参数，--apply 会修改旧批次。本批若触发中断核验，按以下证据框架报告，再由用户决定工具和步骤。

- 核对队列/worker 的 PID 和创建时间是否匹配活进程。旧脚本遇到匹配活进程立即拒绝；runner 退出本身不能证明任务结束，PID 复用也不能算存活证据。
- 无匹配进程，但 motion.npz 或 worker_result.json 存在：旧脚本停止并要求检查部分或完整产物，文件存在不等于成功。
- 无匹配进程、上述两类文件均不存在：旧脚本结合 process_started 事件、资源记录和系统启动时间形成中断证据；仅在 --apply 时写入中断记录，同身份时转 pending，不同身份时创建恢复计划。本批不直接执行这些写操作。
- integrity.json 缺失可能只是生成阶段尚未进入 contract，不能据此判失败；done=true 且所列文件哈希一致，并有成功状态证据时，可报告为“可能已完成但队列状态未落盘”，仍不自行改状态；未完成、哈希不符或证据冲突则报告。
- PRIMAL 需额外列出 factorial/ 产物的存在情况。旧脚本只检查 motion.npz/worker_result.json，不覆盖所有本批方法，也不检查 integrity，不能把旧脚本通过等同契约通过。
- 参考证据限本批状态、进程身份、事件时间、文件存在性、必要状态字段及 integrity/hash；不读取动作数组、指标、留出清单输出。

## 结束条件和交付

analysis=done、阶段 returncode=0，且 analysis/development.json 存在，三项同时满足才报告完成；仅检查存在性，不打开结果。报告阶段耗时、按任务 ID 合并重试后的最终 success/failed/timeout、开发清单 missing 的任务、重试后仍 resource_gate 的任务、各评估阶段未评估行数和 errors、两盘最终可用空间。彩排复用与历史复用另列，self 与 lumbar 共享目录不重复合计。missing 只从开发清单的 metadata 读取，不打开留出清单。

结束后检查本笔记不含地址、凭据或指标，仅显式提交本笔记；被 git 忽略时仅对本文件 add -f，commit 限定本文件，不夹带其他会话内容。不提交采样脚本、测试、采样日志或其他文件。确认采样器已自行退出后结束管理。

## 2026-11-10T00:40:57+08:00 首次监控

当前阶段 `generate_main`；runner_alive=true。数据盘 1035.871 GiB，系统盘 58.413 GiB，系统盘相对启动基线 -0.000107 GiB；free -g available=18，精确可用内存 18.817 GiB。评估尚未开始，RSS 不适用；evalkit evaluated/errors 尚无对应行。采样器无 ALERT。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 4 | 0 | 0 | 0 | 448 | 2 |
| omnicontrol | 0 | 0 | 0 | 0 | 908 | 0 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

其他阶段均 pending；CHOIS 队列尚未建立，其 84 个计划任务不伪装成 status 已报告计数。下一次定时采样预计在 01:40:57 左右（Asia/Shanghai），由实际采样耗时决定。

## 2026-11-10T01:41:38+08:00 每小时监控

当前阶段 `generate_main`；runner_alive=true。采样器 PID 1672558 已核实仍存活，独立日志已有 2 行，最近采样 2026-11-10T01:40:57+08:00。

数据盘 1112034963456 bytes（1035.663 GiB）；系统盘 62717571072 bytes（58.410 GiB）；系统盘相对启动基线 -2564096 bytes（-0.002388 GiB）。free -g available=16，精确 available=16.804 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 22 | 0 | 0 | 0 | 884 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进，failed/timeout/resource_gate 均为 0；未触发告警，按计划继续监控。

## 2026-11-10T02:42:51+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 3 行，最近采样 2026-11-10T02:40:57+08:00。

数据盘 1111806672896 bytes（1035.451 GiB）；系统盘 62715658240 bytes（58.409 GiB）；系统盘相对启动基线 -4476928 bytes（-0.004169 GiB）。free -g available=16，精确 available=16.735 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 109 | 0 | 0 | 0 | 797 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T03:41:12+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 4 行，最近采样 2026-11-10T03:40:57+08:00。

数据盘 1111588966400 bytes（1035.248 GiB）；系统盘 62707015680 bytes（58.400 GiB）；系统盘相对启动基线 -13119488 bytes（-0.012218 GiB）。free -g available=16，精确 available=16.730 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 193 | 0 | 0 | 0 | 713 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T04:41:13+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 5 行，最近采样 2026-11-10T04:40:57+08:00。

数据盘 1111363739648 bytes（1035.038 GiB）；系统盘 62758080512 bytes（58.448 GiB）；系统盘相对启动基线 37945344 bytes（+0.035339 GiB）。free -g available=16，精确 available=16.768 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 279 | 0 | 0 | 0 | 627 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T05:40:59+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 6 行，最近采样 2026-11-10T05:40:58+08:00。

数据盘 1111138238464 bytes（1034.828 GiB）；系统盘 62728019968 bytes（58.420 GiB）；系统盘相对启动基线 7884800 bytes（+0.007343 GiB）。free -g available=17，精确 available=17.480 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 366 | 0 | 0 | 0 | 540 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T06:41:24+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 7 行，最近采样 2026-11-10T06:40:58+08:00。

数据盘 1110912942080 bytes（1034.618 GiB）；系统盘 62702317568 bytes（58.396 GiB）；系统盘相对启动基线 -17817600 bytes（-0.016594 GiB）。free -g available=16，精确 available=16.493 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 452 | 0 | 0 | 0 | 454 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T07:41:10+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 8 行，最近采样 2026-11-10T07:40:58+08:00。

数据盘 1110688317440 bytes（1034.409 GiB）；系统盘 62700773376 bytes（58.395 GiB）；系统盘相对启动基线 -19361792 bytes（-0.018032 GiB）。free -g available=16，精确 available=16.651 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 538 | 0 | 0 | 0 | 368 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T08:41:14+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 9 行，最近采样 2026-11-10T08:40:58+08:00。

数据盘 1110457311232 bytes（1034.194 GiB）；系统盘 62699712512 bytes（58.394 GiB）；系统盘相对启动基线 -20422656 bytes（-0.019020 GiB）。free -g available=16，精确 available=16.565 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 626 | 0 | 0 | 0 | 280 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T09:41:25+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 10 行，最近采样 2026-11-10T09:40:58+08:00。

数据盘 1110221553664 bytes（1033.974 GiB）；系统盘 62689595392 bytes（58.384 GiB）；系统盘相对启动基线 -30539776 bytes（-0.028442 GiB）。free -g available=16，精确 available=16.451 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 717 | 0 | 0 | 0 | 189 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T10:41:13+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 11 行，最近采样 2026-11-10T10:40:58+08:00。

数据盘 1109988679680 bytes（1033.758 GiB）；系统盘 62687690752 bytes（58.382 GiB）；系统盘相对启动基线 -32444416 bytes（-0.030216 GiB）。free -g available=16，精确 available=16.502 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 806 | 0 | 0 | 0 | 100 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T11:41:09+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 12 行，最近采样 2026-11-10T11:40:58+08:00。

数据盘 1109754392576 bytes（1033.539 GiB）；系统盘 62685528064 bytes（58.380 GiB）；系统盘相对启动基线 -34607104 bytes（-0.032230 GiB）。free -g available=16，精确 available=16.564 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 896 | 0 | 0 | 0 | 10 | 2 |
| closd | 0 | 0 | 0 | 0 | 238 | 0 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T12:41:14+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 13 行，最近采样 2026-11-10T12:40:58+08:00。

数据盘 1109543600128 bytes（1033.343 GiB）；系统盘 62687641600 bytes（58.382 GiB）；系统盘相对启动基线 -32493568 bytes（-0.030262 GiB）。free -g available=11，精确 available=11.420 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 137 | 0 | 0 | 0 | 99 | 2 |
| dart | 0 | 0 | 0 | 0 | 238 | 0 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T13:41:12+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 14 行，最近采样 2026-11-10T13:40:58+08:00。

数据盘 1109287469056 bytes（1033.104 GiB）；系统盘 62677401600 bytes（58.373 GiB）；系统盘相对启动基线 -42733568 bytes（-0.039799 GiB）。free -g available=16，精确 available=16.213 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 142 | 0 | 0 | 0 | 94 | 2 |
| ardy | 0 | 0 | 0 | 0 | 120 | 0 |
| primal | 0 | 0 | 0 | 0 | 59 | 0 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T14:41:15+08:00 每小时监控

当前阶段 generate_main；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 15 行，最近采样 2026-11-10T14:40:59+08:00。

数据盘 1108833751040 bytes（1032.682 GiB）；系统盘 62675492864 bytes（58.371 GiB）；系统盘相对启动基线 -44642304 bytes（-0.041576 GiB）。free -g available=17，精确 available=17.488 GiB。

generate_main：running。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 40 | 0 | 0 | 0 | 17 | 2 |

评估尚未开始，evaluated/errors 与评估 worker RSS 不适用。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T15:41:22+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 16 行，最近采样 2026-11-10T15:40:59+08:00。

数据盘 1102274220032 bytes（1026.573 GiB）；系统盘 62671642624 bytes（58.368 GiB）；系统盘相对启动基线 -48492544 bytes（-0.045162 GiB）。free -g available=18，精确 available=18.422 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 38, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 1661524, "max_worker_rss_kib": 214556, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T16:41:15+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 17 行，最近采样 2026-11-10T16:40:59+08:00。

数据盘 1096320397312 bytes（1021.028 GiB）；系统盘 62669107200 bytes（58.365 GiB）；系统盘相对启动基线 -51027968 bytes（-0.047523 GiB）。free -g available=18，精确 available=18.432 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 38, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 1674024, "max_worker_rss_kib": 222192, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["no_counter_progress_over_hour"]

### 2026-11-10T16:41:59+08:00 计数一小时未变化的只读排查

触发规则：eval_method 的 evaluated=38、errors=0 一小时未变，runner_alive=true。只检查了 nvidia-smi、属于本批的进程状态、stages.json 的阶段时间/状态字段以及已冻结的编排代码；未读取评估日志、评估输出或分析结果。

- 两张 GPU 仅有桌面应用，无计算任务。评估父进程存在，运行器在等待它；8 个评估 worker 均为 RNl，CPU 占用采样为 108%–129%，进程年龄为 7–21 秒。评估阶段从 15:24:54 开始，这些最近创建且活跃的 worker 支持进程池仍在处理任务的判断。
- 生成队列均 finished，其 heartbeat 停在生成结束时间是正常状态；评估阶段没有独立的周期性 heartbeat，不能将旧生成 heartbeat 当作评估停滞证据。
- 冻结代码 evalkit_batch.py 的 run_batch 先为全部清单行执行 offset_job（拟合/偏移前置处理），之后才执行写入 evalkit.json 的 job；status 的 evaluated 不统计前置处理完成行数。该评估入口不使用生成队列的 GPU 共享锁。
- 判断：当前不是等待其他会话 GPU 锁；与仍在前置处理的情况一致，这是依据进程活动和代码顺序的推断，未读取输出确认具体前置进度。errors=0，资源无告警。按该条处置规则报告，继续只读监控，不做任何进程或状态修改。

## 2026-11-10T17:41:24+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 18 行，最近采样 2026-11-10T17:40:59+08:00。

数据盘 1088658522112 bytes（1013.892 GiB）；系统盘 62657265664 bytes（58.354 GiB）；系统盘相对启动基线 -62869504 bytes（-0.058552 GiB）。free -g available=18，精确 available=18.300 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 38, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 7, "total_rss_kib": 1428724, "max_worker_rss_kib": 231680, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["no_counter_progress_over_hour"]

### 2026-11-10T17:42:11+08:00 计数未变化的进程复查

阶段 eval_method，runner_alive=True，evaluated/errors={"rows": 3592, "evaluated": 38, "errors": 0}。GPU 计算进程数=0；本批评估父进程等待状态=['futex_wait_queue']。

| worker PID | 状态 | 进程年龄（秒） | CPU 时间 | CPU 占用（%） | RSS（KiB） | 等待状态 |
|---:|---|---:|---|---:|---:|---|
| 2696720 | RNl | 8 | 00:00:09 | 121.0 | 231540 | - |
| 2696741 | RNl | 7 | 00:00:08 | 123.0 | 206948 | - |
| 2696762 | RNl | 7 | 00:00:08 | 119.0 | 206932 | - |
| 2696782 | RNl | 7 | 00:00:08 | 121.0 | 216716 | - |
| 2696822 | RNl | 6 | 00:00:07 | 126.0 | 207044 | - |
| 2696881 | RNl | 5 | 00:00:06 | 128.0 | 205984 | - |
| 2696905 | RNl | 4 | 00:00:05 | 142.0 | 205944 | - |
| 2697023 | RNl | 0 | 00:00:01 | 0.0 | 157192 | - |

这里只记录操作系统进程证据；不把前置处理推断当作已评估行数。未读取评估日志或输出，未修改任何进程或状态。

## 2026-11-10T18:41:13+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 19 行，最近采样 2026-11-10T18:41:00+08:00。

数据盘 1085532786688 bytes（1010.981 GiB）；系统盘 62654365696 bytes（58.351 GiB）；系统盘相对启动基线 -65769472 bytes（-0.061253 GiB）。free -g available=17，精确 available=17.969 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 409, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2173844, "max_worker_rss_kib": 272432, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T19:41:12+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 20 行，最近采样 2026-11-10T19:41:00+08:00。

数据盘 1083516321792 bytes（1009.103 GiB）；系统盘 62644400128 bytes（58.342 GiB）；系统盘相对启动基线 -75735040 bytes（-0.070534 GiB）。free -g available=17，精确 available=17.903 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 623, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2081048, "max_worker_rss_kib": 271908, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T20:48:16+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 21 行，最近采样 2026-11-10T20:41:01+08:00。

数据盘 1081326280704 bytes（1007.064 GiB）；系统盘 62633652224 bytes（58.332 GiB）；系统盘相对启动基线 -86482944 bytes（-0.080544 GiB）。free -g available=17，精确 available=17.862 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 855, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2170836, "max_worker_rss_kib": 273012, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T21:41:21+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 22 行，最近采样 2026-11-10T21:41:01+08:00。

数据盘 1079804813312 bytes（1005.647 GiB）；系统盘 62632099840 bytes（58.331 GiB）；系统盘相对启动基线 -88035328 bytes（-0.081989 GiB）。free -g available=17，精确 available=17.600 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 998, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2482916, "max_worker_rss_kib": 311924, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T22:41:23+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 23 行，最近采样 2026-11-10T22:41:01+08:00。

数据盘 1077883977728 bytes（1003.858 GiB）；系统盘 62630129664 bytes（58.329 GiB）；系统盘相对启动基线 -90005504 bytes（-0.083824 GiB）。free -g available=17，精确 available=17.565 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1099, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2394116, "max_worker_rss_kib": 310608, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-10T23:41:21+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 24 行，最近采样 2026-11-10T23:41:02+08:00。

数据盘 1076040634368 bytes（1002.141 GiB）；系统盘 62628093952 bytes（58.327 GiB）；系统盘相对启动基线 -92041216 bytes（-0.085720 GiB）。free -g available=17，精确 available=17.546 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1196, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2469536, "max_worker_rss_kib": 310300, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T00:41:25+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 25 行，最近采样 2026-11-11T00:41:02+08:00。

数据盘 1073824038912 bytes（1000.077 GiB）；系统盘 62448238592 bytes（58.159 GiB）；系统盘相对启动基线 -271896576 bytes（-0.253223 GiB）。free -g available=16，精确 available=16.917 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1293, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2407896, "max_worker_rss_kib": 311836, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T01:41:21+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 26 行，最近采样 2026-11-11T01:41:03+08:00。

数据盘 1068251189248 bytes（994.886 GiB）；系统盘 62082695168 bytes（57.819 GiB）；系统盘相对启动基线 -637440000 bytes（-0.593662 GiB）。free -g available=16，精确 available=16.207 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1390, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2404792, "max_worker_rss_kib": 311760, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T02:41:20+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 27 行，最近采样 2026-11-11T02:41:03+08:00。

数据盘 1065602265088 bytes（992.419 GiB）；系统盘 55612137472 bytes（51.793 GiB）；系统盘相对启动基线 -7107997696 bytes（-6.619839 GiB）。free -g available=16，精确 available=16.449 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1486, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2398252, "max_worker_rss_kib": 310884, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

### 2026-11-11T02:45:49+08:00 — 系统盘空间告警的只读定位

02:41:20 检查触发 `system_available_drop_over_5_gib`：系统盘可用 55,612,137,472 bytes，比启动基线 62,720,135,168 bytes 少 7,107,997,696 bytes（约 6.62 GiB）。阈值 5 GiB 为约定，未经推演。后续复测可用 55,611,183,104 bytes，`free -g` available=16。运行器存活，eval_method evaluated=1486/3592、errors=0；未处置任何进程或文件。

按规则执行 `timeout 120 du -xsh /tmp ~/.cache`：/tmp 显示 1.9G，~/.cache 显示 35G。部分系统私有临时目录无读取权限、一个既有挂载端点不可访问；仅记录覆盖限制，未提权。缓存目录进一步按字节统计为 36,703,731,712 bytes。这些是当前总量，不能作为相对启动时的增长量。

批次路径仍为符号链接并解析到 data_root，`findmnt -T` 显示数据盘 /dev/sda1；系统盘为 /dev/nvme0n1p3。采样时识别出本批运行器/评估及其后代共 10 个进程；其 20 个普通文件描述符均指向数据盘，未发现系统盘普通文件描述符。这是瞬时证据，不能排除已关闭文件的历史写入。

只读目录占用与 mtime/ctime 元数据定位显示：01:41:21 之后有变化的 ~/Downloads 文件共 6 个，当前分配字节合计 6,979,694,592；~/.config/Cursor 有变化的文件当前合计 26,884,026,368 bytes。未读取这些文件内容。前者与本小时系统盘空间下降量接近，但缺少目录级历史基线，无法据此确定增长来源；后者同样不是增长量。/tmp 与缓存中同期变化文件的当前大小不足以直接解释本次下降。

结论：尚未发现本批输出写错盘的证据；系统盘告警已报告用户，未清理、未改配置、未改状态、未停止或重启运行器。继续既定的只读采样与小时记录，不进行自动修复。下次检查继续记录系统盘相对启动基线的变化。

## 2026-11-11T03:41:18+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 28 行，最近采样 2026-11-11T03:41:04+08:00。

数据盘 1063789305856 bytes（990.731 GiB）；系统盘 56989536256 bytes（53.076 GiB）；系统盘相对启动基线 -5730598912 bytes（-5.337036 GiB）。free -g available=16，精确 available=16.411 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1581, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2329348, "max_worker_rss_kib": 311296, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

2026-11-11T03:41:37+08:00 系统盘告警续记：03:41 系统盘可用 56,989,536,256 bytes，比启动基线少 5,730,598,912 bytes（约 5.34 GiB），较 02:41 回升 1,377,398,784 bytes（约 1.28 GiB）。仍超过 5 GiB 阈值（约定，未经推演）。再次按原命令只读定位：[{"location": "/tmp", "du_h_size": "1.9G"}, {"location": "~/.cache", "du_h_size": "35G"}]；du 返回码 1，覆盖限制同前。未清理、未处置运行器；没有新增的本批输出写错盘证据。

2026-11-11T03:58:39+08:00 证据口径更正：02:45 定位中“运行器/评估及其后代共 10 个”的匹配实际覆盖评估进程树，没有覆盖内部子命令为 `_run` 的编排运行器。现按准确的 `_run --batch` 身份重新只读检查运行器及全部后代：{"timestamp": "2026-11-11T03:58:39+08:00", "matched_runner_pids": [1671665], "own_process_count": 11, "regular_open_fds_by_location": {"data": 24}, "system_open_files": [], "metadata_errors": 0}。此前数据盘判断仍成立，但上述进程范围以此次更正为准；文件描述符证据仍仅是瞬时采样，未读取输出内容。

## 2026-11-11T04:41:16+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 29 行，最近采样 2026-11-11T04:41:04+08:00。

数据盘 1061897994240 bytes（988.970 GiB）；系统盘 56987357184 bytes（53.074 GiB）；系统盘相对启动基线 -5732777984 bytes（-5.339066 GiB）。free -g available=16，精确 available=16.317 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1680, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2473340, "max_worker_rss_kib": 311288, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T05:41:21+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 30 行，最近采样 2026-11-11T05:41:04+08:00。

数据盘 1059933999104 bytes（987.140 GiB）；系统盘 56987176960 bytes（53.073 GiB）；系统盘相对启动基线 -5732958208 bytes（-5.339233 GiB）。free -g available=16，精确 available=16.257 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1824, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2478540, "max_worker_rss_kib": 311652, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T06:41:23+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 31 行，最近采样 2026-11-11T06:41:05+08:00。

数据盘 1057997807616 bytes（985.337 GiB）；系统盘 56947286016 bytes（53.036 GiB）；系统盘相对启动基线 -5772849152 bytes（-5.376385 GiB）。free -g available=16，精确 available=16.352 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 1926, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2473284, "max_worker_rss_kib": 310280, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T07:41:22+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 32 行，最近采样 2026-11-11T07:41:05+08:00。

数据盘 1056178270208 bytes（983.643 GiB）；系统盘 56944541696 bytes（53.034 GiB）；系统盘相对启动基线 -5775593472 bytes（-5.378941 GiB）。free -g available=16，精确 available=16.271 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2022, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2480276, "max_worker_rss_kib": 311572, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T08:41:35+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 33 行，最近采样 2026-11-11T08:41:06+08:00。

数据盘 1054205915136 bytes（981.806 GiB）；系统盘 56941273088 bytes（53.031 GiB）；系统盘相对启动基线 -5778862080 bytes（-5.381985 GiB）。free -g available=16，精确 available=16.154 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2126, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2462100, "max_worker_rss_kib": 309144, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T09:41:45+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 34 行，最近采样 2026-11-11T09:41:06+08:00。

数据盘 1052298285056 bytes（980.029 GiB）；系统盘 56940630016 bytes（53.030 GiB）；系统盘相对启动基线 -5779505152 bytes（-5.382584 GiB）。free -g available=16，精确 available=16.337 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2226, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2469860, "max_worker_rss_kib": 311340, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T10:41:42+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 35 行，最近采样 2026-11-11T10:41:06+08:00。

数据盘 1050408140800 bytes（978.269 GiB）；系统盘 56940683264 bytes（53.030 GiB）；系统盘相对启动基线 -5779451904 bytes（-5.382534 GiB）。free -g available=16，精确 available=16.291 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2325, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2485212, "max_worker_rss_kib": 311432, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T11:41:39+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 36 行，最近采样 2026-11-11T11:41:07+08:00。

数据盘 1048426934272 bytes（976.424 GiB）；系统盘 56938831872 bytes（53.028 GiB）；系统盘相对启动基线 -5781303296 bytes（-5.384258 GiB）。free -g available=16，精确 available=16.531 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2429, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 1963912, "max_worker_rss_kib": 307504, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T12:41:41+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 37 行，最近采样 2026-11-11T12:41:07+08:00。

数据盘 1046533226496 bytes（974.660 GiB）；系统盘 56931860480 bytes（53.022 GiB）；系统盘相对启动基线 -5788274688 bytes（-5.390751 GiB）。free -g available=16，精确 available=16.356 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2528, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 2400328, "max_worker_rss_kib": 311412, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T13:41:38+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 38 行，最近采样 2026-11-11T13:41:07+08:00。

数据盘 1044604809216 bytes（972.864 GiB）；系统盘 56928505856 bytes（53.019 GiB）；系统盘相对启动基线 -5791629312 bytes（-5.393875 GiB）。free -g available=16，精确 available=16.576 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 2709, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 1985660, "max_worker_rss_kib": 251408, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

## 2026-11-11T14:41:35+08:00 每小时监控

当前阶段 eval_method；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 39 行，最近采样 2026-11-11T14:41:07+08:00。

数据盘 1036716929024 bytes（965.518 GiB）；系统盘 59709243392 bytes（55.609 GiB）；系统盘相对启动基线 -3010891776 bytes（-2.804111 GiB）。free -g available=16，精确 available=16.704 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3323, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_method", "workers": 8, "total_rss_kib": 1860336, "max_worker_rss_kib": 247172, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

2026-11-11T14:41:35+08:00 系统盘告警解除：可用 59,709,243,392 bytes，较启动基线少 3,010,891,776 bytes（约 2.80 GiB），已低于减少超过 5 GiB 的告警阈值（约定，未经推演）。较上一小时回升 2,780,737,536 bytes；本 agent 未删除或清理任何文件，也未处置其他会话。回升来源未核实，不作归因。eval_method evaluated=3323/3592、errors=0，runner 与采样器正常；继续既定小时监控。

## 2026-11-11T15:41:59+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 40 行，最近采样 2026-11-11T15:41:08+08:00。

数据盘 1030241214464 bytes（959.487 GiB）；系统盘 57664192512 bytes（53.704 GiB）；系统盘相对启动基线 -5055942656 bytes（-4.708714 GiB）。free -g available=16，精确 available=16.844 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 38, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 1603928, "max_worker_rss_kib": 235244, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T16:41:39+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 41 行，最近采样 2026-11-11T16:41:08+08:00。

数据盘 1024152576000 bytes（953.816 GiB）；系统盘 57652490240 bytes（53.693 GiB）；系统盘相对启动基线 -5067644928 bytes（-4.719612 GiB）。free -g available=16，精确 available=16.857 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 38, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 1667020, "max_worker_rss_kib": 208520, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["no_counter_progress_over_hour"]

### 2026-11-11T16:41:51+08:00 计数未变化的进程复查

阶段 eval_self，runner_alive=True，evaluated/errors={"rows": 3592, "evaluated": 38, "errors": 0}。GPU 计算进程数=0；本批评估父进程等待状态=['futex_wait_queue']。

| worker PID | 状态 | 进程年龄（秒） | CPU 时间 | CPU 占用（%） | RSS（KiB） | 等待状态 |
|---:|---|---:|---|---:|---:|---|
| 2994206 | RNl | 30 | 00:00:31 | 105.0 | 217500 | - |
| 2994226 | RNl | 28 | 00:00:30 | 108.0 | 208348 | - |
| 2994290 | RNl | 26 | 00:00:27 | 106.0 | 208272 | - |
| 2994330 | RNl | 22 | 00:00:24 | 109.0 | 208392 | - |
| 2994381 | RNl | 17 | 00:00:18 | 110.0 | 208360 | - |
| 2994445 | RNl | 9 | 00:00:11 | 122.0 | 208428 | - |
| 2994465 | RNl | 9 | 00:00:10 | 121.0 | 208380 | - |
| 2994530 | RNl | 4 | 00:00:06 | 162.0 | 208400 | - |

这里只记录操作系统进程证据；不把前置处理推断当作已评估行数。未读取评估日志或输出，未修改任何进程或状态。

### 2026-11-11T16:42:55+08:00 计数未变化的进程复查

阶段 eval_self，runner_alive=True，evaluated/errors={"rows": 3592, "evaluated": 38, "errors": 0}。GPU 计算进程数=0；本批评估父进程等待状态=['futex_wait_queue']。

| worker PID | 状态 | 进程年龄（秒） | CPU 时间 | CPU 占用（%） | RSS（KiB） | 等待状态 |
|---:|---|---:|---|---:|---:|---|
| 2994862 | RNl | 47 | 00:00:47 | 102.0 | 235880 | - |
| 2995162 | RNl | 20 | 00:00:21 | 105.0 | 208224 | - |
| 2995182 | RNl | 19 | 00:00:20 | 106.0 | 208444 | - |
| 2995247 | RNl | 14 | 00:00:15 | 107.0 | 208552 | - |
| 2995303 | RNl | 8 | 00:00:08 | 111.0 | 208424 | - |
| 2995366 | RNl | 5 | 00:00:05 | 119.0 | 208564 | - |
| 2995411 | RNl | 2 | 00:00:02 | 121.0 | 166716 | - |
| 2995431 | RNl | 2 | 00:00:02 | 124.0 | 166800 | - |

这里只记录操作系统进程证据；不把前置处理推断当作已评估行数。未读取评估日志或输出，未修改任何进程或状态。

2026-11-11T16:42:55+08:00 eval_self 计数未推进的判断：两次只读进程检查（16:41:51、16:42:55）均有 8 个活跃 worker，状态 RNl，CPU 使用约 102%–162%；两次 worker PID 全部更替，进程年龄 2–47 秒。GPU 无计算进程，评估父进程在 futex_wait_queue 等待。已完成生成阶段的 heartbeat 保留完成时间；评估阶段没有独立 heartbeat。上述进程证据未显示等待共享 GPU 锁。结合先前已读的 evalkit_batch.py 全量 offset/拟合预处理先于写入 evaluated 计数的行为，当前情况与预处理仍在推进一致（代码与进程证据推断，未读取日志或评估输出，不能据此给出预处理完成行数）。已报告用户，保持既定小时监控；未重启、未终止、未改状态。

## 2026-11-11T17:41:37+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 42 行，最近采样 2026-11-11T17:41:09+08:00。

数据盘 1017016426496 bytes（947.170 GiB）；系统盘 57657053184 bytes（53.697 GiB）；系统盘相对启动基线 -5063081984 bytes（-4.715363 GiB）。free -g available=16，精确 available=16.430 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 120, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2016820, "max_worker_rss_kib": 283208, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T18:41:39+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 43 行，最近采样 2026-11-11T18:41:09+08:00。

数据盘 1014985981952 bytes（945.279 GiB）；系统盘 57654947840 bytes（53.695 GiB）；系统盘相对启动基线 -5065187328 bytes（-4.717323 GiB）。free -g available=16，精确 available=16.378 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 466, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2179876, "max_worker_rss_kib": 277604, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T19:41:38+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 44 行，最近采样 2026-11-11T19:41:10+08:00。

数据盘 1012999708672 bytes（943.429 GiB）；系统盘 57653223424 bytes（53.694 GiB）；系统盘相对启动基线 -5066911744 bytes（-4.718929 GiB）。free -g available=16，精确 available=16.260 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 677, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2168684, "max_worker_rss_kib": 272100, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T20:45:21+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 45 行，最近采样 2026-11-11T20:41:10+08:00。

数据盘 1010977497088 bytes（941.546 GiB）；系统盘 57643139072 bytes（53.684 GiB）；系统盘相对启动基线 -5076996096 bytes（-4.728321 GiB）。free -g available=16，精确 available=16.261 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 891, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2136968, "max_worker_rss_kib": 272556, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

20:45:21 本次人工检查较计划的 20:41 延后约 4 分钟，按实际时间记录；后台采样器最近一条记录时间为 2026-11-11T20:41:10+08:00，当前阶段 eval_self。未补写虚构的人工检查时间。

## 2026-11-11T21:41:39+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 46 行，最近采样 2026-11-11T21:41:11+08:00。

数据盘 1009375010816 bytes（940.054 GiB）；系统盘 57641562112 bytes（53.683 GiB）；系统盘相对启动基线 -5078573056 bytes（-4.729790 GiB）。free -g available=16，精确 available=16.008 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1020, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2470276, "max_worker_rss_kib": 310700, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T22:41:31+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 47 行，最近采样 2026-11-11T22:41:11+08:00。

数据盘 1007549419520 bytes（938.354 GiB）；系统盘 57639518208 bytes（53.681 GiB）；系统盘相对启动基线 -5080616960 bytes（-4.731693 GiB）。free -g available=16，精确 available=16.015 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1116, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2477140, "max_worker_rss_kib": 310764, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-11T23:41:37+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 48 行，最近采样 2026-11-11T23:41:12+08:00。

数据盘 1005688823808 bytes（936.621 GiB）；系统盘 57637494784 bytes（53.679 GiB）；系统盘相对启动基线 -5082640384 bytes（-4.733578 GiB）。free -g available=15，精确 available=15.989 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1214, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2410108, "max_worker_rss_kib": 311620, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T00:41:36+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 49 行，最近采样 2026-11-12T00:41:12+08:00。

数据盘 1003773468672 bytes（934.837 GiB）；系统盘 56151003136 bytes（52.295 GiB）；系统盘相对启动基线 -6569132032 bytes（-6.117981 GiB）。free -g available=16，精确 available=16.023 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1315, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2325828, "max_worker_rss_kib": 311728, "measurement": "instantaneous_ps_rss"}。

告警/待处置：["system_available_drop_over_5_gib"]

### 2026-11-12T00:42:07+08:00 — 系统盘告警只读定位

{"timestamp": "2026-11-12T00:42:07+08:00", "du_usage_not_growth": [{"location": "/tmp", "bytes": 1933852672}, {"location": "~/.cache", "bytes": 36708450304}], "du_returncode": 1, "du_diagnostic_line_count": 11, "matched_runner_pids": [1671665], "own_process_count": 11, "regular_open_fds_by_location": {"data": 24}, "system_open_files": [], "metadata_errors": 0, "batch_is_symlink": true, "batch_resolves_under_data_root": true, "batch_device": "/dev/sda1", "system_device": "/dev/nvme0n1p3"}

目录大小是当前占用，未作为增长量；部分 /tmp 私有目录权限/挂载覆盖限制以诊断行数记录，不提权。文件描述符与挂载检查是瞬时证据，不能排除历史已关闭文件。未读取其他会话文件内容、评估输出或日志；未删除、未清理、未处置进程，继续只读监控。

## 2026-11-12T01:41:38+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 50 行，最近采样 2026-11-12T01:41:13+08:00。

数据盘 996863406080 bytes（928.401 GiB）；系统盘 61359345664 bytes（57.145 GiB）；系统盘相对启动基线 -1360789504 bytes（-1.267334 GiB）。free -g available=15，精确 available=15.523 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1412, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2463916, "max_worker_rss_kib": 310496, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

2026-11-12T01:41:38+08:00 系统盘告警再次解除：可用 61,359,345,664 bytes，较启动基线少 1,360,789,504 bytes（约 1.27 GiB），低于减少超过 5 GiB 的告警阈值（约定，未经推演）。较 00:41 回升 5,208,342,528 bytes。本 agent 未删除、清理或移动任何实验/其他会话文件；空间回升来源未核实，不作归因。eval_self evaluated=1412/3592、errors=0，继续既定监控。

## 2026-11-12T02:41:28+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 51 行，最近采样 2026-11-12T02:41:13+08:00。

数据盘 995077292032 bytes（926.738 GiB）；系统盘 61355724800 bytes（57.142 GiB）；系统盘相对启动基线 -1364410368 bytes（-1.270706 GiB）。free -g available=15，精确 available=15.437 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1506, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2328900, "max_worker_rss_kib": 310460, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T03:41:36+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 52 行，最近采样 2026-11-12T03:41:14+08:00。

数据盘 993309704192 bytes（925.092 GiB）；系统盘 61354803200 bytes（57.141 GiB）；系统盘相对启动基线 -1365331968 bytes（-1.271564 GiB）。free -g available=15，精确 available=15.447 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1598, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2624372, "max_worker_rss_kib": 457004, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T04:41:36+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 53 行，最近采样 2026-11-12T04:41:14+08:00。

数据盘 991423758336 bytes（923.335 GiB）；系统盘 61350506496 bytes（57.137 GiB）；系统盘相对启动基线 -1369628672 bytes（-1.275566 GiB）。free -g available=15，精确 available=15.625 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1698, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2467252, "max_worker_rss_kib": 311540, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T05:41:31+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 54 行，最近采样 2026-11-12T05:41:15+08:00。

数据盘 989633060864 bytes（921.668 GiB）；系统盘 61348245504 bytes（57.135 GiB）；系统盘相对启动基线 -1371889664 bytes（-1.277672 GiB）。free -g available=15，精确 available=15.659 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1833, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2395660, "max_worker_rss_kib": 309748, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T06:41:34+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 55 行，最近采样 2026-11-12T06:41:15+08:00。

数据盘 987774586880 bytes（919.937 GiB）；系统盘 61328048128 bytes（57.116 GiB）；系统盘相对启动基线 -1392087040 bytes（-1.296482 GiB）。free -g available=15，精确 available=15.668 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 1931, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2404624, "max_worker_rss_kib": 311448, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T07:41:36+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 56 行，最近采样 2026-11-12T07:41:16+08:00。

数据盘 986034360320 bytes（918.316 GiB）；系统盘 61325824000 bytes（57.114 GiB）；系统盘相对启动基线 -1394311168 bytes（-1.298553 GiB）。free -g available=15，精确 available=15.713 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2023, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2359912, "max_worker_rss_kib": 311912, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T08:41:42+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 57 行，最近采样 2026-11-12T08:41:16+08:00。

数据盘 984138543104 bytes（916.550 GiB）；系统盘 61329260544 bytes（57.117 GiB）；系统盘相对启动基线 -1390874624 bytes（-1.295353 GiB）。free -g available=15，精确 available=15.660 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2123, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2399148, "max_worker_rss_kib": 311056, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T09:41:37+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 58 行，最近采样 2026-11-12T09:41:17+08:00。

数据盘 982408810496 bytes（914.940 GiB）；系统盘 61317419008 bytes（57.106 GiB）；系统盘相对启动基线 -1402716160 bytes（-1.306381 GiB）。free -g available=15，精确 available=15.664 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2214, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2284076, "max_worker_rss_kib": 311192, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T10:41:44+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 59 行，最近采样 2026-11-12T10:41:17+08:00。

数据盘 980598448128 bytes（913.253 GiB）；系统盘 61314330624 bytes（57.103 GiB）；系统盘相对启动基线 -1405804544 bytes（-1.309258 GiB）。free -g available=15，精确 available=15.180 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2309, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2477916, "max_worker_rss_kib": 311548, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T11:41:37+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 60 行，最近采样 2026-11-12T11:41:18+08:00。

数据盘 978733502464 bytes（911.517 GiB）；系统盘 61312028672 bytes（57.101 GiB）；系统盘相对启动基线 -1408106496 bytes（-1.311401 GiB）。free -g available=15，精确 available=15.864 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2407, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2337580, "max_worker_rss_kib": 311512, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T12:41:42+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 61 行，最近采样 2026-11-12T12:41:18+08:00。

数据盘 976829755392 bytes（909.744 GiB）；系统盘 61310291968 bytes（57.100 GiB）；系统盘相对启动基线 -1409843200 bytes（-1.313019 GiB）。free -g available=15，精确 available=15.760 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2507, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2470184, "max_worker_rss_kib": 310812, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T13:41:39+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 62 行，最近采样 2026-11-12T13:41:19+08:00。

数据盘 974987472896 bytes（908.028 GiB）；系统盘 61307060224 bytes（57.097 GiB）；系统盘相对启动基线 -1413074944 bytes（-1.316029 GiB）。free -g available=15，精确 available=15.743 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 2604, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 2476524, "max_worker_rss_kib": 311240, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T14:41:32+08:00 每小时监控

当前阶段 eval_self；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 63 行，最近采样 2026-11-12T14:41:19+08:00。

数据盘 973087854592 bytes（906.259 GiB）；系统盘 61303918592 bytes（57.094 GiB）；系统盘相对启动基线 -1416216576 bytes（-1.318954 GiB）。free -g available=16，精确 available=16.238 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3099, "errors": 0}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_self", "workers": 8, "total_rss_kib": 1910968, "max_worker_rss_kib": 255508, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-12T15:41:35+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 64 行，最近采样 2026-11-12T15:41:20+08:00。

数据盘 971164196864 bytes（904.467 GiB）；系统盘 61292818432 bytes（57.083 GiB）；系统盘相对启动基线 -1427316736 bytes（-1.329292 GiB）。free -g available=15，精确 available=15.093 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3591, "errors": 1}。

eval_lumbar：{"rows": 624, "evaluated": 19, "errors": 1}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3087444, "max_worker_rss_kib": 433160, "measurement": "instantaneous_ps_rss"}。

首次观察到 error 的阶段（等待允许的 rg 诊断）：eval_self, eval_lumbar。

告警/待处置：["eval_lumbar: errors >2% evaluated"]

### 2026-11-12T15:42:49+08:00 — 首次评估 error 与待用户决定事项

15:41:35 状态：eval_method done，evaluated=3592/3592、errors=0；eval_self done，evaluated=3591/3592、errors=1；eval_lumbar running，evaluated=19/624、errors=1。运行器存活。腰部行按约定分母 evaluated 计算的错误比例为 1/19≈5.26%，超过 2% 告警阈值（约定，未经推演）。self/lumbar 共享目录，两个 errors=1 不能直接相加为两个独立错误；此处不推断是否同一行。

严格按允许范围执行以下两条命令：

```bash
PYTHONNOUSERSITE=1 rg -n "Traceback|Error" develop_logs/test_runs/20261110_batch_main/logs/eval_self.log
PYTHONNOUSERSITE=1 rg -n "Traceback|Error" develop_logs/test_runs/20261110_batch_main/logs/eval_lumbar.log
```

两条命令输出均为空，返回码均为 1（无匹配），不是报错内容被省略。结论：无法据此判断 error 原因及零星/系统性性质。未打开任何 evalkit.json、error.json、完整评估日志或分析结果，未扩大读取范围。

按确认计划“无匹配时记录‘无法据此判断’并询问，不扩大读取范围”，报告并通过 AskQuestion 等待用户决定后续监控/核验方式。停止自动处置；存活运行器保持运行，采样器继续既定只读记录，不 kill、不重跑、不改状态或配置。15:41 的内存 available=16,206,024,704 bytes（free -g available=15），腰部 8 个 worker 瞬时总 RSS=3,087,444 KiB、单进程最大=433,160 KiB；当次未触发内存或磁盘告警。

## 2026-11-12T22:40:51+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 70 行，最近采样 2026-11-12T21:41:24+08:00。

数据盘 965474807808 bytes（899.168 GiB）；系统盘 61259284480 bytes（57.052 GiB）；系统盘相对启动基线 -1460850688 bytes（-1.360523 GiB）。free -g available=14，精确 available=14.012 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 359, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 4313912, "max_worker_rss_kib": 548772, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

### 2026-11-12T22:41:41+08:00 — 用户决定与等待期间记录补记

用户已选择“继续原定的每小时只读监控，保留错误，结束时汇总”。恢复 agent 小时检查，不重跑、不扩大结果读取范围。此前因等待用户决定而停止主动检查，后台采样器持续运行。以下为后台采样日志的补记，明确不是同期人工检查。已核实采样器 PID 1672558 的进程身份存活，并用 `rg -n "^ALERT " develop_logs/notes/20261110_batch_monitor_samples.log` 汇总该时段，未见采样器资源/存活告警。

16:41–21:41 的腰部 evaluated 持续推进；21:41 共享目录中 self/lumbar 的 errors 均由 1 变为 2，self 的状态计数同时由 evaluated=3591 变为 3590。仅记录 status 计数变化，不将两个 errors=2 相加、不推断具体错误原因。单小时新增为 1，累计腰部 errors/evaluated 也未超过 2%（告警阈值为约定，未经推演）；保留错误，继续已获授权的监控。

#### 后台采样 2026-11-12T16:41:21+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3591/3592，errors=1。

eval_lumbar：evaluated=116/624，errors=1。

数据盘可用 970119065600 bytes；系统盘可用 61289426944 bytes；系统盘相对启动变化 -1430708224 bytes；内存 available=16248430592 bytes；free -g available=15。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3139956, "max_worker_rss_kib": 438680, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

#### 后台采样 2026-11-12T17:41:22+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3591/3592，errors=1。

eval_lumbar：evaluated=183/624，errors=1。

数据盘可用 969210736640 bytes；系统盘可用 61277597696 bytes；系统盘相对启动变化 -1442537472 bytes；内存 available=16731975680 bytes；free -g available=15。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 2690688, "max_worker_rss_kib": 460032, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

#### 后台采样 2026-11-12T18:41:22+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3591/3592，errors=1。

eval_lumbar：evaluated=234/624，errors=1。

数据盘可用 968379564032 bytes；系统盘可用 61275422720 bytes；系统盘相对启动变化 -1444712448 bytes；内存 available=15826780160 bytes；free -g available=14。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3652148, "max_worker_rss_kib": 537252, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

#### 后台采样 2026-11-12T19:41:23+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3591/3592，errors=1。

eval_lumbar：evaluated=265/624，errors=1。

数据盘可用 967657877504 bytes；系统盘可用 61273223168 bytes；系统盘相对启动变化 -1446912000 bytes；内存 available=15647686656 bytes；free -g available=14。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3754340, "max_worker_rss_kib": 548964, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

#### 后台采样 2026-11-12T20:41:24+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3591/3592，errors=1。

eval_lumbar：evaluated=297/624，errors=1。

数据盘可用 966927896576 bytes；系统盘可用 61271597056 bytes；系统盘相对启动变化 -1448538112 bytes；内存 available=15472828416 bytes；free -g available=14。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3751292, "max_worker_rss_kib": 548884, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

#### 后台采样 2026-11-12T21:41:24+08:00（恢复后补记）

当前阶段 eval_lumbar；runner_alive=True。
| 阶段 | 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---|---:|---:|---:|---:|---:|---:|
| generate_main | motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| generate_main | omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| generate_main | closd | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | dart | 238 | 0 | 0 | 0 | 0 | 0 |
| generate_main | ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| generate_main | primal | 59 | 0 | 0 | 0 | 0 | 0 |
| generate_chois | chois | 84 | 0 | 0 | 0 | 0 | 0 |

阶段状态：generate_main=done, generate_chois=done, generate_retry=done, contract=done, manifest=done, seed_evalkit=done, eval_method=done, eval_self=done, eval_lumbar=running, neck_rotation=pending, analysis=pending。

eval_method：evaluated=3592/3592，errors=0。

eval_self：evaluated=3590/3592，errors=2。

eval_lumbar：evaluated=329/624，errors=2。

数据盘可用 966188195840 bytes；系统盘可用 61262753792 bytes；系统盘相对启动变化 -1457381376 bytes；内存 available=15606267904 bytes；free -g available=14。

评估 worker RSS（瞬时采样，不是真实峰值）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3867904, "max_worker_rss_kib": 538008, "measurement": "instantaneous_ps_rss"}。

采样器 ALERT：[]。

2026-11-12T22:43:04+08:00 结束汇总口径核对（仅读源码）：batch_pipeline.py 的 eval_counts 根据清单行逐项检查 error.json 是否存在，优先计 errors；再判断 evalkit 存在与腰部维度标记。status 对已 done 的评估阶段也重新计算当前目录计数。eval_self 与 eval_lumbar 共用 development/self，故腰部执行期间 self 的实时 evaluated/errors 仍可变化，不能将两个阶段当前 errors 直接相加；stages.json 中阶段结束时的 counts 与实时 status 应区分。此次仅阅读源码，未打开任何评估结果文件。最终耗时/退出码从 stages.json 的 seconds/returncode 等状态字段取得，analysis/development.json 仅检查存在性，missing 仅读开发清单的 missing 元数据。

## 2026-11-12T23:41:50+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 72 行，最近采样 2026-11-12T23:41:25+08:00。

数据盘 964648071168 bytes（898.399 GiB）；系统盘 61252489216 bytes（57.046 GiB）；系统盘相对启动基线 -1467645952 bytes（-1.366852 GiB）。free -g available=14，精确 available=14.264 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 419, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3580828, "max_worker_rss_kib": 548768, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T00:42:15+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 73 行，最近采样 2026-11-13T00:41:26+08:00。

数据盘 963994378240 bytes（897.790 GiB）；系统盘 61262118912 bytes（57.055 GiB）；系统盘相对启动基线 -1458016256 bytes（-1.357883 GiB）。free -g available=14，精确 available=14.219 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 448, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3886304, "max_worker_rss_kib": 549016, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T01:42:01+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 74 行，最近采样 2026-11-13T01:41:27+08:00。

数据盘 963349053440 bytes（897.189 GiB）；系统盘 61259059200 bytes（57.052 GiB）；系统盘相对启动基线 -1461075968 bytes（-1.360733 GiB）。free -g available=13，精确 available=13.783 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 489, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 4290436, "max_worker_rss_kib": 537244, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T02:41:45+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 75 行，最近采样 2026-11-13T02:41:27+08:00。

数据盘 962610192384 bytes（896.501 GiB）；系统盘 61304614912 bytes（57.094 GiB）；系统盘相对启动基线 -1415520256 bytes（-1.318306 GiB）。free -g available=13，精确 available=13.788 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 521, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 4288644, "max_worker_rss_kib": 539052, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T03:41:36+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 76 行，最近采样 2026-11-13T03:41:28+08:00。

数据盘 961871548416 bytes（895.813 GiB）；系统盘 61292023808 bytes（57.083 GiB）；系统盘相对启动基线 -1428111360 bytes（-1.330032 GiB）。free -g available=14，精确 available=14.112 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3590, "errors": 2}。

eval_lumbar：{"rows": 624, "evaluated": 553, "errors": 2}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 3918604, "max_worker_rss_kib": 536376, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T04:41:59+08:00 每小时监控

当前阶段 eval_lumbar；runner_alive=true。采样器 PID 1672558，存活=True；独立日志已有 77 行，最近采样 2026-11-13T04:41:29+08:00。

数据盘 961072668672 bytes（895.069 GiB）；系统盘 61114974208 bytes（56.918 GiB）；系统盘相对启动基线 -1605160960 bytes（-1.494923 GiB）。free -g available=15，精确 available=15.194 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3589, "errors": 3}。

eval_lumbar：{"rows": 624, "evaluated": 614, "errors": 3}。

评估 RSS（KiB，瞬时采样）：{"stage": "eval_lumbar", "workers": 8, "total_rss_kib": 2630500, "max_worker_rss_kib": 428416, "measurement": "instantaneous_ps_rss"}。

计数正常推进；未触发告警，按计划继续监控。

## 2026-11-13T05:41:51+08:00 每小时监控

当前阶段 completed；runner_alive=false。采样器 PID 1672558，存活=False；独立日志已有 79 行，最近采样 2026-11-13T05:41:33+08:00。

数据盘 960974688256 bytes（894.977 GiB）；系统盘 61104136192 bytes（56.908 GiB）；系统盘相对启动基线 -1615998976 bytes（-1.505016 GiB）。free -g available=17，精确 available=17.005 GiB。

generate_main：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| motionlcm | 454 | 0 | 0 | 0 | 0 | 0 |
| omnicontrol | 908 | 0 | 0 | 0 | 0 | 0 |
| closd | 238 | 0 | 0 | 0 | 0 | 0 |
| dart | 238 | 0 | 0 | 0 | 0 | 0 |
| ardy | 120 | 0 | 0 | 0 | 0 | 0 |
| primal | 59 | 0 | 0 | 0 | 0 | 0 |

generate_chois：done。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| chois | 84 | 0 | 0 | 0 | 0 | 0 |

eval_method：{"rows": 3592, "evaluated": 3592, "errors": 0}。

eval_self：{"rows": 3592, "evaluated": 3588, "errors": 4}。

eval_lumbar：{"rows": 624, "evaluated": 620, "errors": 4}。

当前没有运行中的 eval_method/eval_self/eval_lumbar，评估 worker RSS 不适用。

告警/待处置：["runner_not_alive"]

### 2026-11-13T05:45:28+08:00 — 正式批次结束验收与管理关闭

批次 `20261110_batch_main` 按冻结编排完成。开始时间 2026-11-10T00:40:17+08:00，结束时间 2026-11-13T05:07:10+08:00，实测总历时 76 小时 26 分 53 秒。全部阶段为 done，returncode 均为 0；这表示编排正常结束，不表示每个评估行都成功。保留评估错误，不重跑、不替换、不改状态。

| 阶段 | 状态 | 退出码 | 记录耗时（时:分:秒） |
|---|---|---:|---:|
| generate_main | done | 0 | 14:14:30 |
| generate_chois | done | 0 | 00:28:10 |
| generate_retry | done | 0 | 00:00:00 |
| contract | done | 0 | 00:00:58 |
| manifest | done | 0 | 00:00:18 |
| seed_evalkit | done | 0 | 00:00:42 |
| eval_method | done | 0 | 23:41:30 |
| eval_self | done | 0 | 24:20:43 |
| eval_lumbar | done | 0 | 13:19:08 |
| neck_rotation | done | 0 | 00:17:34 |
| analysis | done | 0 | 00:03:22 |

耗时采用 stages.json 的累计 seconds 字段，脚本按秒舍入，个别阶段与开始/结束时间之差可相差 1 秒。总历时按开始与结束时间相减，以上均为实测记录，不是约定耗时。

生成队列按任务 ID 合并去重（main、chois、retry 顺序），新任务共 2101 个。queue_main 与 queue_chois 均 finished；无 queue_retry 状态文件，此批无剩余资源准入拒绝任务，无需该重试队列。

| 方法 | success | failed | timeout | resource_gate | pending | running |
|---|---:|---:|---:|---:|---:|---:|
| MotionLCM | 454 | 0 | 0 | 0 | 0 | 0 |
| OmniControl | 908 | 0 | 0 | 0 | 0 | 0 |
| CLoSD | 238 | 0 | 0 | 0 | 0 | 0 |
| DART | 238 | 0 | 0 | 0 | 0 | 0 |
| ARDY | 120 | 0 | 0 | 0 | 0 | 0 |
| PRIMAL | 59 | 0 | 0 | 0 | 0 | 0 |
| CHOIS | 84 | 0 | 0 | 0 | 0 | 0 |
| 合计 | 2101 | 0 | 0 | 0 | 0 | 0 |

彩排复用 15 个（main 13、CHOIS 2）与冻结历史复用 644 个另列，未混入上述新任务计数。计数依据启动自检与队列最终状态；生成最终状态不能用于声称内部从未发生 timeout 尝试。

仅读开发清单 `evalkit/manifest_development.json` 的 missing 元数据：`[]`，没有 missing 任务。合并后的剩余 resource_gate 任务清单：`[]`。未打开留出清单及其输出，也未对留出清单进行评估。

| 评估阶段 | rows | evaluated（最终实时状态） | 未评估行数（rows − evaluated） | errors |
|---|---:|---:|---:|---:|
| eval_method | 3592 | 3592 | 0 | 0 |
| eval_self | 3592 | 3588 | 4 | 4 |
| eval_lumbar | 624 | 620 | 4 | 4 |

self 与 lumbar 共用 development/self。仅从这两份开发清单的行 ID 与 error.json 文件存在性核对：两组错误行集合相同，交集 4、并集 4。因此是同一组 4 个错误行，不能相加为 8；腰部为 self 的子集。未读取任何 error.json 内容或指标数值。阶段执行结束时保存的 counts 与上述最终实时目录计数含义不同，以该表说明最终状态。首条错误的允许 rg 检查无匹配，之后按用户明确决定继续监控、保留错误；原因未作推断。

最终磁盘可用空间（2026-11-13T05:42:43+08:00 复核）：

- 数据盘：960,974,688,256 bytes，约 894.98 GiB。
- 系统盘：61,104,058,368 bytes，约 56.91 GiB；比启动基线少 1,616,076,800 bytes，约 1.51 GiB。

analysis 为 done 且 returncode=0；仅确认 `analysis/development.json` 是现存文件，没有打开内容。是否及何时查看开发集结果由用户决定。

运行器已正常结束。采样器 PID 1672558 已不再存活；日志最后两条采样分别为 2026-11-13T05:41:29+08:00 与 05:41:33+08:00，均显示 completed、analysis done、runner_alive=false，确认按约定追加最终采样后自行退出。此处 runner_not_alive 是正常结束状态，不是待 resume 的中断。PID 文件与所有采样/运行产物原样保留，不清理。

交接笔记已检查地址/凭据特征及科学指标数值特征，未发现匹配；内容限于运行状态、任务/错误计数、资源、进程证据和管理记录。未改配置或编排计划，未碰其他会话进程与文件。最终仅提交本笔记，其他已有修改、未跟踪文件、采样日志及采样器代码均不纳入提交。管理工作在本笔记提交并核对后结束，不再定时检查。
