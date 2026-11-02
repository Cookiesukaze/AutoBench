# AutoBench

视频测评顺序为：计划 → 基础画质测量与三路判定 → 正式画质 → 事件/物理判定 → 三维等权计分。基础画质的问题率单独保存，不换算为正式画质分。

```text
autobench/
  app/
    cli.py                  # 命令行入口
    run.py                  # 阶段调用与成本记录
    config.py               # 四份配置校验
    data/
      types.py              # 数据契约
      store.py              # 产物路径、原子写入、成本记录
    batch/
      run.py                # 批次编排、并发判定、失败隔离
      inputs.py             # 视频清单、身份与内容摘要
      state.py              # 独占锁、任务历史、续跑校验
    remote/
      ssh.py                # SSH 与流式上传下载
      stage.py              # 代码快照、远端作业与回收
      collect.py            # 产物完整性校验
      gpu/
        runner.py           # 逐卡显存预约、专家依赖与常驻进程
        worker.py           # 专家队列、权重校验、原始观测
  stages/
    plan/                   # run.py + steps/
    basic_quality/
      run.py
      steps/
        step01_measure.py
        step02_three_routes.py
        step03_gate.py
      experts/              # 亮度、闪烁、失真、运动、抖动、HADM、MUSIQ、DOVER
    quality/                # run.py + steps/ + experts/
    judge/                  # 保留 steps/、experts/、tools/
    score/                  # 保留 steps/、audit/
  prompt/
config/                     # checks.json / vlm.json / paths.json / gpus.json
tests/
```

## 使用

在仓库根目录安装：`python -m pip install -e .`。也可以直接运行 `python -m autobench.app.cli`。本地依赖支持 Python 3.10+；GPU 模型依赖沿用远端各专家的独立环境，不合并安装进本地环境。

视频清单必须明确配置与完整 unit，路径相对于清单文件：

```json
{
  "T1": {"Q_E1__seed48": "videos/T1.mp4"},
  "T10": {"Q_E1__seed48": "videos/T10.mp4"}
}
```

题面为 `{"tasks": [{"question_id": "Q_E1", "image_sha256": "...", "spec": {...}}]}`。参考图必须为 `<image-dir>/<question_id>.jpg`，摘要必须和题面一致。

```powershell
# CPU 测量、基础三路、正式画质；不发起 VLM 请求
python -m autobench.app.cli batch --run-id demo --specs tasks.json --image-dir images --videos videos.json

# 同一批次继续正式判定与计分；此选项会发起配置的 VLM API 请求
python -m autobench.app.cli batch --run-id demo --specs tasks.json --image-dir images --videos videos.json --judge

# GPU 专家上传、调度、下载；可加 --judge 完成正式判定
python -m autobench.app.cli batch --run-id gpu-demo --specs tasks.json --image-dir images --videos videos.json --remote
```

`--videos-dir` 可替代 `--videos`，发现规则为 `<config>/<question>/seed<N>/<run>/*.mp4`；同配置同 unit 多个候选会报错。`--parents parents.json` 声明原版关系，例如 `{"T10": "T1"}`。所有子命令支持 `--config-dir`。其他独立入口：`plan`、`basic-quality`、`quality`、`judge`、`score`，具体参数见 `--help`。

`--quality-dir` 导入 `<unit>/summary.json`，其中必须包含 `unit` 和 `inputs: {配置: 视频SHA256}`。`--experts-dir` 导入 `<config>_<unit>/`，必须带 `identity.json`，包含 `config`、`unit`、`video_sha256`。旧归档身份不全时会报错，不能通过相似题号静默拼接。导入目录不能与本批次产物目录重叠。

## 判定与计分

基础检测保留三个独立证据：绝对阈值、同题同 seed 与原版比较、同题同 seed 与配置组比较。绝对路命中即可告警；两条相对路同时命中告警；相机抖动、低运动允许组内一路告警。全部缺失时保留未测量。组内过半绝对异常时取消组内参照。亮度台阶单独贴标签，不进入问题概率；HADM 按置信度和连续异常时长判定。

第一关沿用旧版当前入口的准入规则：有测量的样本继续正式评测，完全未测量的样本阻塞，告警留作诊断。没有擅自启用 `config_threshold` 硬淘汰配置。

正式总分固定为 `(事件遵循分 + 物理分 + 正式画质分) / 3`。画质配方写在 `checks.quality_score`：

- 默认迁移旧版清晰度的**题内百分位**：同题同 seed 比较，最低 0、最高 1、同值同分；单个有效配置时保留未定。这个默认是迁移假设，尚未由用户最终确认。
- `method: percentage` 支持把专家原始值按显式 `range: [下界, 上界]` 换算到 [0,1]。超出范围报错，缺失保留未定。
- 正式画质从不读取第一关 `rates`，也不使用 `1 - 问题率`。

缺少视频、任何正式维度缺失或判定未定，配置总分保留 `null` 并退出排名。排行榜点估计与 `score.json` 使用同一个总分；bootstrap 按题重采样，也使用真实的三维视频分。题集不同的配置不直接排名。

## 续跑和远端边界

产物保存在 `paths.local.data_root/runs_subdir/run_id/`。`manifest.json` 固定输入、参考图、源码与配置摘要；`batch.json` 保留阶段状态和历史。重跑同一命令仅复用内容摘要仍匹配的成功任务。输入、代码或配置变化时必须使用新 run-id。锁文件记录 PID，异常中断遗留的 `batch.lock` 需核对原进程退出后移除。

远端代码上传至 `archive/autobench/<run-id>/<identity>/code`，不会覆盖旧 AutoInferOpt 源码。专家源码、环境和已安装权重继续从 `paths.remote` 指定位置读取。SAM 后构造跟踪点；每种专家用一个常驻进程消费整个批次；调度器按物理 GPU UUID 和逐卡显存预算预约，退出后才归还预约。下载产物逐项 SHA256 校验。断线后同命令重新连接；部分失败的专家再次运行时可复用完整且校验成功的观测。

CPU 模式测清晰度、亮度、闪烁和生成失真。MUSIQ、HADM、SAM、CoTracker、WAFT、DINO、姿态模型需要对应远端环境和权重。MUSIQ 已核对并使用 tracker 环境和已有 KonIQ 权重，显式传入权重路径及 SHA256，避免隐式下载。DOVER 适配器已接入，但 `paths.remote.dover_repo` 默认 `null`；指向含 `dover_batch.py` 和 `dover.yml` 的已安装目录才启用。Ghosting 仍标记为未接线。缺失通道不会被当作正常。DINO 当前只提供视频全帧相似度，没有冒充参考图主体身份识别。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m ruff check autobench tests --select F
python -m compileall -q autobench tests
```

测试使用真实合成视频、脚本判官和模拟 GPU 子进程，覆盖批次执行与续跑、三路、连续 HADM、缺失计分、排行榜和产物完整性。另已通过 SSH 只读核对 A800 上三个解释器、五份专家 manifest 及对应源码/模型目录、HADM 源码与权重均存在。本次迁移没有启动真实远端模型或付费 VLM 调用；远端模型兼容性与实际显存仍需真实作业验证。
