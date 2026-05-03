# 超车数据处理与建模流水线规划

超车条线文档与脚本归档根目录为 **`overtaking/`**（与 `following/` 并列）。相位检测、合规切段等历史脚本当前仍在 [`following/scripts/extract_overtaking_phases.py`](../following/scripts/extract_overtaking_phases.py)、[`following/scripts/crop_overtaking_segments.py`](../following/scripts/crop_overtaking_segments.py)，本目录提供 IL 清洗、按风格训练包装、聚类与生成入口。

---

## 1. 现状盘点（仓库内）

| 环节 | 跟驰（已有） | 超车（已有 / 本仓库新增） |
|------|-------------|---------------------------|
| 原始数据 | `following` / `*_f` / `exp*_f` | `exp[123]_o`，见 `extract_overtaking_phases._discover_overtaking_csvs` |
| 合规切段 | — | `crop_overtaking_segments.py` → 整段 `driving_data.csv` 树 |
| IL 切段清洗 | `clean_following_for_imitation.py` → `segment_*.csv` | **`overtaking/scripts/clean_overtaking_for_imitation.py`** |
| BC 训练 | `following/train/train_bc_gru.py` | 同上；**`overtaking/train/train_bc_overtaking_by_style.py`** 包装多风格 |
| 风格聚类 | `cluster_following_style.py` | **`overtaking/scripts/cluster_overtaking_style.py`** |
| 生成 | `generate_no_driver_following_outputs.py` | **`overtaking/train/generate_no_driver_overtaking_outputs.py`**（MVP：仅替换纵向，横向保持场景 CSV） |

---

## 2. 与跟驰的本质差异

1. **场景语义**：跟驰为右车道匀速跟车；超车在 **前车达目标速（reach）后** 经历 **右→左→回右**，相对几何与 `distance_headway` / `ttc` 的有效解释窗口与跟驰不同。
2. **校准**：`calibrate_following_data.py` 将 `ego_pos_y` 压向右车道中心，**不应用于超车**（会破坏变道轨迹）。若需纵向去噪，应单独设计（本流水线默认在 **crop 后原始 y** 上清洗）。
3. **清洗**：跟驰的「前车长期无效则丢段」对变道可能过严；超车清洗脚本默认 **`--skip_meaningful_lead_check`**（可关闭以沿用跟驰逻辑）。
4. **实验分层**：`exp1_o` / `exp2_o` / `exp3_o` 对应 35/50/65 km/h；训练时可分速验证或联合训练。
5. **聚类特征**：跟驰侧重间距、时距、纵加；超车脚本侧重 **左车道占比时间、`|ego_pos_y|` 跨道范围、横向速度分位、thw/ttc 极小值、纵加峰值** 等。
6. **生成（MVP）**：跟驰可「公共前车 + 横向残差池」；超车 MVP 为 **前车/世界列不变 + 仅纵向用 GRU 预测并积分**，**`ego_pos_y` / `ego_yaw` / `steer` 保持场景 CSV 原样**（后续可改为联合横纵模型或参考轨迹混合）。

---

## 3. 推荐命令流水线

### 3.1 切段（可选，推荐）

从原始 `data/` 生成合规超车窗口（与跟驰数据目录结构镜像）：

```bash
python3 following/scripts/crop_overtaking_segments.py \
  --data_dir data \
  --out_dir overtaking/outputs/overtaking_cropped
```

### 3.2 IL 清洗 → `segment_*.csv`

输入可为 **crop 输出目录** 或含 `exp*_o` 的原始树：

```bash
python3 overtaking/scripts/clean_overtaking_for_imitation.py \
  --data_dir overtaking/outputs/overtaking_cropped \
  --out_dir overtaking/outputs/overtaking_il_clean_gap04 \
  --gap_threshold_sec 0.4 \
  --min_segment_duration_sec 5.0
```

（默认**跳过**跟驰用的「前车有效」筛段；若要与跟驰一致，加 `--enforce_meaningful_lead`。）

### 3.3 按司机训练 BC-GRU（示例）

```bash
python3 following/train/train_bc_gru.py \
  --data_dir overtaking/outputs/overtaking_il_clean_gap04/T9 \
  --out_dir overtaking/outputs/il_bc_gru_per_driver/T9_longitudinal_framewin \
  --train_drivers T9 --val_drivers T9 --test_drivers T9 \
  --split_within_driver \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
  --seq_len 20 --epochs 60
```

### 3.4 风格聚类（≥3 名司机）

```bash
python3 overtaking/scripts/cluster_overtaking_style.py \
  --data_dir overtaking/outputs/overtaking_il_clean_gap04 \
  --out_dir overtaking/outputs/overtaking_style_clusters \
  --plot --seed 42
```

### 3.5 按风格训练

```bash
python3 overtaking/train/train_bc_overtaking_by_style.py \
  --conservative T2,T9 --neutral T7,T10 --aggressive T3,T5 \
  --data_dir overtaking/outputs/overtaking_il_clean_gap04 \
  --out_root overtaking/outputs/il_bc_gru_by_style
```

### 3.6 典型场景生成（MVP）

```bash
python3 overtaking/train/generate_typical_overtaking_by_style.py \
  --conservative T2,T9 --neutral T7,T10 --aggressive T3,T5 \
  --model_root overtaking/outputs/il_bc_gru_by_style \
  --common_case_dir data/T12/行车/.../exp1_o/... \
  --out_root overtaking/outputs/typical_overtaking_by_style \
  --seed 42 --warmup_frames 20
```

（将 `common_case_dir` 设为某一司机某次 **`driving_data.csv` 所在会话目录**，使所有风格在同一超车场景下对比纵向策略。）

---

## 4. 数据范围说明

- `extract_overtaking_phases._discover_overtaking_csvs` **默认排除** `pre_familiarization`；若需练习数据，需改发现规则或在清洗 `--data_dir` 中仅放入正式实验树。
- 车道带参数（`left_y_*` / `right_y_*`）与相位脚本一致；聚类脚本使用相同默认边界做 **左/右车道示性** 统计。

---

## 5. 风险与开放问题

- **样本量**：超车事件/司机数可能少于跟驰，k-means k=3 需至少 3 名司机。
- **清洗输入**：默认推荐 **crop 后** 再清洗，避免全长录制中无关片段；若从全长 `exp*_o` 清洗，需自行保证与 crop 合规逻辑一致。
- **生成 Phase2**：联合横纵或「参考横向 + 学习纵向」需扩展 `model_meta` 与生成脚本。

---

## 6. 文件索引

| 路径 | 作用 |
|------|------|
| `overtaking/scripts/clean_overtaking_for_imitation.py` | 超车 IL 清洗 |
| `overtaking/scripts/cluster_overtaking_style.py` | 超车风格聚类 |
| `overtaking/train/train_bc_overtaking_by_style.py` | 按风格调用 `train_bc_gru.py` |
| `overtaking/train/generate_no_driver_overtaking_outputs.py` | 单模型超车场景纵向生成（MVP） |
| `overtaking/train/generate_typical_overtaking_by_style.py` | 按风格批量调用生成脚本 |
