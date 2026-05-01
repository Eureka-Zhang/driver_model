# 跟驰（Following）场景：数据清洗、模型训练与无驾驶员数据生成

本文档说明 `following/` 目录下跟驰实验的推荐流水线、各脚本职责、**输入 / 输出 / 损失权重**以及生成阶段的物理含义。路径以仓库根目录 `driver_model/` 为基准时可写相对路径。

---

## 1. 总览流水线

推荐顺序：

1. **轨迹校准（可选但推荐）**  
   `scripts/calibrate_following_data.py`  
   对原始 `driving_data.csv` 做纵向加速度清洗、横向 `ego_pos_y` 向车道中心收缩等，**不修改**前车与车间距相关列。

2. **模仿学习专用清洗**  
   `scripts/clean_following_for_imitation.py`  
   在校准结果（或原始数据）上切段、去异常、打 `ttc`/`time_headway` 有效位，输出 `segment_XXX.csv`。

3. **行为克隆训练（BC-GRU）**  
   `train/train_bc_gru.py`  
   读取所有 `segment_*.csv`，按序列窗口训练 GRU，默认预测纵向加速度 `ego_a_long`。

4. **无驾驶员纵向生成 + 横向回放**  
   `train/generate_no_driver_following_outputs.py`  
   加载训练好的模型，在**保持前车轨迹与世界状态不变**的前提下，用模型替换自车纵向加速度，再积分得到 `ego_pos_x`/`ego_speed`，横向按策略叠加“残差抖动”。

可选扩展：

- **跟驰风格聚类**：`scripts/cluster_following_style.py`
- **按风格池化多人数据训练**：`scripts/train_bc_following_by_style.py`（内部调用 `train_bc_gru.py`）

---

## 2. 实验与数据命名（业务约定）

与主数据说明一致时，可作筛选与标注依据：

| 后缀 / 标记 | 含义 |
|-------------|------|
| `_f` | 跟驰实验 |
| `_o` | 超车实验（跟驰流水线会排除） |
| `_b` | 坏数据（清洗脚本路径中含 `_b` 会跳过） |
| `_h` | 半可用数据（当前脚本未自动截半，需自行规则或后处理） |
| `exp1` / `exp2` / `exp3` | 前车基准速度约 **35 / 50 / 65 km/h**（具体以实验设计为准） |

**路径匹配注意**：`clean_following_for_imitation.py` 与早期 `calibrate_following_data.py` 的“发现规则”不完全相同——校准脚本已支持 `..._exp1|exp2|exp3` 目录；清洗脚本主要认路径中的 `following` 或目录名匹配 `*_f`。若你的会话目录仅为 `..._exp1` 而无 `_f`，需在清洗发现逻辑中与校准对齐，或统一将文件夹命名为 `..._exp1_f` 等形式。

---

## 3. 数据清洗

### 3.1 `scripts/calibrate_following_data.py`（校准）

**目的**：在保留专家记录与控制量的前提下，减轻模拟器噪声、不熟悉操作带来的不合理纵向加速度，并适度拉回横向。

**原则（摘自脚本注释）**：

- **专家动作 / 自车状态**（如 `throttle`, `brake`, `steer`, `ego_speed`, `ego_yaw` 等）：默认**不随意改写**；纵向对 `ego_acceleration` 做中值滤波、滑动平均与裁剪后，可写回并推导 `ego_jerk`。
- **前车与跟驰情景列**（`lead_*`, `distance_headway`, `time_headway`, `relative_speed`, `ttc` 等）：**保持不变**（仅在校准流程中按约定重算分解速度时可增加列）。
- **横向**：`ego_pos_y` 向右侧车道中心线收缩（默认 `y_center=-7.625` m，`lateral_scale` 默认 0.5），并可对校准后的 `y` 做滑动平均。

**主要参数**：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--data_dir` | （见脚本） | 含 `driving_data.csv` 的根目录（可按司机子目录只跑单人） |
| `--out_dir` | （见脚本） | 输出目录，相对输入保持镜像结构 |
| `--y_center` | -7.625 | 右车道中心线 `ego_pos_y`（m） |
| `--lateral_scale` | 0.5 | 相对中心的横向偏移缩放 |
| `--y_smooth_window` | 9 | 校准后 `ego_pos_y` 滑动平均窗口 |
| `--acc_median_window` | 5 | 加速度中值滤波窗口 |
| `--acc_smooth_window` | 7 | 加速度滑动平均窗口 |
| `--acc_clip_min` / `--acc_clip_max` | -8 / 6 | 加速度裁剪（m/s²） |
| `--kinematics_mode` | preserve | `preserve` 保留原 `ego_speed` 等；`recompute` 由校准路径重算运动学 |
| `--steer_mode` | copy | `copy` / `bicycle` / `zero` |

**示例（仅 T9）**：

```bash
python3 following/scripts/calibrate_following_data.py \
  --data_dir data/T9 \
  --out_dir following/outputs/following_calibrated/T9
```

---

### 3.2 `scripts/clean_following_for_imitation.py`（模仿学习切段）

**目的**：得到**时间连续**、**前有有效交互**的片段，供序列模型使用；**不**对专家油门/刹车/转向等做平滑或插值，只**删行**或**切段**。

**流程摘要**：

1. 丢弃必备字段缺失行（`timestamp`, `ego_pos_*`, `ego_speed`, `ego_acceleration`, `throttle`, `brake`, `steer`, `lead_*`, `distance_headway` 等）。
2. 按 `timestamp` 排序、去重、保证严格递增。
3. `ttc` / `time_headway` 为 **999** 时视为无效：清空数值并增加 `ttc_valid` / `time_headway_valid`（0/1）。
4. 按相邻帧时间间隔 `> gap_threshold_sec` 切段（默认 0.4 s）。
5. 裁掉头尾长时间“双静止”段（`ego_speed` 与 `lead_speed` 均低于 `v_min_mps`，带 `startup_grace_sec` 宽限）。
6. 丢弃过短段（`min_segment_duration_sec`，默认 5 s）。
7. 在 `|Δego_speed|` 或位置跳变过大处切开并丢弃故障点（`max_speed_jump_mps`, `max_pos_jump_m`）。
8. 若全程车距过大且前车近似不动，丢弃（`max_useful_headway_m`, `v_min_mps`）。

**输出布局**：

```
<out_dir>/T*/行车/<session>/segment_001.csv, segment_002.csv, ...
<out_dir>/cleaning_summary.csv      # 每个保留片段一行
<out_dir>/cleaning_dropped.csv      # 丢弃的源文件及原因
<out_dir>/cleaning_diagnostics.csv  # 每源文件的切段统计
```

**主要参数**：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--data_dir` | `following/outputs/following_calibrated` | 输入根目录（一般为校准后输出） |
| `--out_dir` | `following/outputs/following_il_clean` | 清洗输出根目录 |
| `--gap_threshold_sec` | 0.4 | 超此间隔则新片段 |
| `--min_segment_duration_sec` | 5.0 | 最短片段时长 |
| `--startup_grace_sec` | 2.0 | 头尾静止裁剪宽限 |
| `--v_min_mps` | 0.5 | “静止”速度阈值 |
| `--max_speed_jump_mps` | 6.0 | 速度跳变阈值 |
| `--max_pos_jump_m` | 4.0 | 位置跳变阈值 |
| `--max_useful_headway_m` | 200.0 | 判定是否有前车交互 |

**示例**：

```bash
python3 following/scripts/clean_following_for_imitation.py \
  --data_dir following/outputs/following_calibrated/T9 \
  --out_dir following/outputs/following_il_clean_gap04/T9 \
  --gap_threshold_sec 0.4
```

---

## 4. 模型训练（`train/train_bc_gru.py`）

### 4.1 数据格式

训练只认形如：

```
<data_dir>/T*/.../segment_*.csv
```

即清洗脚本产出的片段 CSV。司机 ID 从路径中的 `T数字` 解析。

### 4.2 输入特征（默认）

与 `DEFAULT_FEATURES` 一致：

| 特征名 | 说明 |
|--------|------|
| `dt_prev` | 当前帧与上一帧 `timestamp` 差（非均匀采样下显式给出 Δt；首帧为 0） |
| `ego_v_long` | 自车纵向速度（来自列或等价于 `ego_speed` 的解析逻辑） |
| `ego_a_long` | 自车纵向加速度 |
| `distance_headway` | 车头时距对应距离等（原始列） |
| `relative_v_long` | 纵向相对速度（可由 `lead_speed - ego_speed` 等推导） |
| `lead_v_long` | 前车纵向速度 |
| `ttc` | 碰撞时间（无效时在读取逻辑中需与 valid 配合；片段内已部分清洗） |
| `ttc_valid` | 0/1 |
| `time_headway` | 车头时距 |
| `time_headway_valid` | 0/1 |

**标准化**：仅用**训练集**所有时间步展平后计算 `feature_mean` / `feature_std`，再对 train/val/test 同一变换；过小方差维度置为 1.0。

### 4.3 输出目标（默认）

- `DEFAULT_TARGETS = ["ego_a_long"]`  
  即每步预测**当前时刻纵向加速度**（行为克隆常用的动作空间）。

若将来扩展多目标（如同时预测 `throttle`），需同步增加 `target_weights` 长度。

### 4.4 序列与样本构造

- 超参数 `--seq_len`（默认 20）：每个训练样本为长度 `seq_len` 的特征序列，**预测序列最后一帧**对应的 `y`。
- 片段长度 `< seq_len` 的整段丢弃。

### 4.5 损失与权重

- 损失为**加权 MSE**：对 batch 内 `(pred - target)^2` 按目标维乘以权重再平均。
- `--target_weights`：逗号分隔，**与 `targets` 顺序一一对应**。  
  默认仅一个目标：`--target_weights 1.0`。  
  多目标示例：`--target_weights 1.0,0.5`（须与 targets 长度一致）。

### 4.6 划分方式

- **`--split_within_driver`**：在同一批司机内，按**片段路径**随机划分 train/val/test（适合“每人一个模型”）。
- 否则可按 `--train_drivers` / `--val_drivers` / `--test_drivers` 指定不同司机。

比例需满足：`train_ratio + val_ratio + test_ratio = 1.0`（默认 0.7 / 0.15 / 0.15）。

### 4.7 模型与输出文件

- 结构：`BCGRU`（GRU + MLP 头），`hidden_dim`、`num_layers`、`dropout` 可调。
- 早停：`--patience`（验证损失不降则停）。
- 输出目录：
  - `best_model.pt`：最佳验证损失对应的权重
  - `train_report.json`：含 `feature_mean`、`feature_std`、`history`、`test_metrics`、`target_weights` 等
  - `model_meta.json`：`features`、`targets`、`seq_len`、网络超参、`target_weights`（供生成脚本读取）

**每人一模型示例**：

```bash
D=T9
python3 following/train/train_bc_gru.py \
  --data_dir following/outputs/following_il_clean_gap04 \
  --out_dir following/outputs/il_bc_gru_per_driver/${D}_longitudinal_framewin \
  --train_drivers ${D} \
  --val_drivers ${D} \
  --test_drivers ${D} \
  --split_within_driver \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
  --seq_len 20 \
  --target_weights 1.0 \
  --epochs 60
```

---

## 5. 数据生成（`train/generate_no_driver_following_outputs.py`）

### 5.1 目的

在**不改变前车轨迹、车间距、时间轴等记录**的前提下，用训练好的 BC-GRU **替换自车纵向加速度**，再**数值积分**更新 `ego_pos_x` 与纵向速度相关列；横向通过 **`original_jitter` 模式**从某司机的横向残差池中抽样，叠加在 `lane_center_y` 上，得到个性化“车道内抖动”。

### 5.2 输入数据发现

- **场景 CSV**：`discover_following_csvs(data_dir)`，要求目录树中存在 `driving_data.csv`，且路径符合跟驰规则（含 `following` 或 `*_f`，排除 `_b`、`overtaking`、`_o` 等）。  
  即生成阶段读的是**整段会话级** `driving_data.csv`，不是 `segment_*.csv`（与训练输入不同，需注意目录指向）。

### 5.3 模型与归一化

从 `--model_dir` 读取：

- `model_meta.json`：`features`, `targets`, `seq_len`, 网络结构
- `train_report.json`：`feature_mean`, `feature_std`
- `best_model.pt`：权重  

推理时对每个窗口做与训练相同的 `(x - mean) / std`。

### 5.4 纵向逻辑概要

1. 前 `warmup_frames`（默认 20）行：纵向加速度对齐**前车纵向加速度**（`lead_a_long` 或 `lead_acceleration`），便于与场景平滑衔接。
2. 从 `max(warmup_frames, seq_len - 1)` 开始，用 GRU **逐步预测** `targets`（通常为 `ego_a_long`），写回对应列及 `ego_acceleration`（若存在）。
3. 对整段按预测加速度做前向积分，更新 `ego_pos_x`、`ego_speed` / `ego_v_long` 等（保持与脚本内 `_integrate_longitudinal_x` 一致）。

### 5.5 横向逻辑（`--lateral_mode original_jitter`）

- 在 `--lane_center_y`（默认 -7.625）处定义车道中心。
- 从 **横向残差池**采样：`ego_pos_y - lane_center`、以及相对中位数的 `ego_yaw`/`steer` 残差序列；可 `--seed` 控制可复现性。
- `--lateral_pool_data_dir`：从该目录树构建横向池（默认与 `--data_dir` 相同）。会同时扫描 **`segment_*.csv`（模仿学习清洗输出）** 与符合跟驰规则的 **`driving_data.csv`**。
- `--lateral_pool_driver`：池仅保留路径中含该 `T*` 的 CSV；若过滤后为空会告警（见脚本内 `WARN`）。
- `--lane_width`、`--lateral_jitter_limit_ratio`、`--lateral_smooth_window`：对过大残差做平滑/限制（默认超出约 1/4 车道宽时滑动平均）。

### 5.6 输出

- 与输入相对路径一致的 `driving_data.csv` 树结构，写在 `--out_dir`。
- `generation_summary.csv`：每个生成文件一行（行数、预测行数、`driver_id`、横向池来源等）。

**公共前车场景 + 每人纵向模型示例**（脚本顶部注释）：

```bash
COMMON_CASE="/path/to/data/T12/行车/..._exp1_f"
for i in $(seq 1 20); do
  D="T${i}"
  python3 following/train/generate_no_driver_following_outputs.py \
    --data_dir "${COMMON_CASE}" \
    --model_dir "following/outputs/il_bc_gru_per_driver/${D}_longitudinal_framewin" \
    --out_dir "following/outputs/personalized_no_driver_common_lead/${D}" \
    --lateral_mode original_jitter \
    --lane_center_y -7.625 \
    --seed 42
done
```

含义：**场景（前车）固定为 T12 某次实验**；**纵向**仍用每个 `T{i}` 自己的 GRU；**横向抖动**默认也从 `--data_dir` 下发现的数据建池（若目录只有 T12，则横向也来自 T12）。若需横向也个性化，设置 `--lateral_pool_data_dir` / `--lateral_pool_driver`。

---

## 6. 风格聚类与按风格训练（可选）

| 脚本 | 作用 |
|------|------|
| `scripts/cluster_following_style.py` | 根据跟驰特征聚类；`CLUSTER_FEATURES` 含车距、时距、速度/加速度方差及 **加速度/制动强度**（`a>0.2`、`a<-0.2` 下 |a| 的中位数与 75 分位）；k-means 用 `--cluster_dim_weights`（与特征顺序一致，当前 11 维） |
| `scripts/cluster_following_style_leave_one_out.py` | **留一司机法**：每次去掉一名被试后重跑聚类，与全量基线对比标签；输出 `loo_*.csv` 与 `ANALYSIS.md`（至少 4 名司机） |
| `scripts/train_bc_following_by_style.py` | 按 `conservative` / `neutral` / `aggressive` 池化多名司机的 `segment_*.csv`，调用 `train_bc_gru.py` 训练三个风格模型 |

详细参数以各脚本内 `argparse` 为准。

---

## 7. 依赖

- **校准 / 清洗**：标准库 + CSV，无额外硬性依赖。
- **训练与生成**：需要 `numpy`、`torch`（与当前环境一致即可）。

---

## 8. 常见问题

**Q：训练用 `segment_*.csv`，生成为何用 `driving_data.csv`？**  
A：训练需要连续片段与严格切段；生成脚本设计为在**完整会话记录**上回放前车并替换自车纵向，便于与原始实验对齐行数与列。若需只在片段上生成，可将 `data_dir` 指向仅含该会话的目录。

**Q：`target_weights` 多目标怎么用？**  
A：修改 `train_bc_gru.py` 中 `DEFAULT_TARGETS` 并传入与目标个数相同的 `--target_weights`；同时保证清洗后的 CSV 含对应列且非空。

**Q：清洗脚本为何没扫到某些 `exp1` 目录？**  
A：当前 `clean_following_for_imitation.py` 的发现条件为路径含 `following` 或 `*_f`。仅 `exp1` 而无 `_f` 的目录需要与 `calibrate_following_data.py` 的规则统一（在发现函数中增加 `_exp[123]` 或统一文件夹命名）。

---

文档版本与仓库内脚本同步；若脚本默认值变更，以各文件内 `argparse` 与模块常量为准。
