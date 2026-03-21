# URSA-MATH `8eeb664` 中 `vllm/` 的 upstream 来源分析报告

## 1. 分析对象

- 本次分析的目标对象是 `URSA-MATH` 仓库提交 `8eeb664493608516abbf491f1b020b48b7bda013` 里的 `vllm/` 子目录。
- 该提交时间是 `2025-01-22 21:30:32 +0800`。
- 问题不是“这个目录像不像 vLLM”，而是要确认：
  - 它最可能是从 upstream `vllm-project/vllm` 的哪个历史提交 fork 出来的；
  - 它在 upstream 版本线里处于什么位置；
  - 它相对该来源提交到底改了哪些地方。

## 2. 结论

结论先说清楚：

- `8eeb664...` 里的 `vllm/` **不是**某个 upstream tag 的原样复制。
- 它也 **不是** `2025-01-22` 当天 upstream `main` 的直接镜像。
- 它最可能的来源提交是 upstream `vllm-project/vllm` 的：
  - `098f94de42859f8251fe920f87adb88336129c53`
  - 时间：`2024-11-06 14:31:01 +0000`
  - 标题：`[CI/Build] Drop Python 3.8 support (#10038)`
  - 链接：<https://github.com/vllm-project/vllm/commit/098f94de42859f8251fe920f87adb88336129c53>

更准确地说，这份 vendored fork 最像的是：

- `v0.6.3.post1` 之后的 upstream `main`
- `v0.6.4` 之前的 upstream `main`

也就是一份 **pre-`v0.6.4` main 分支快照**，后续再被 Bytedance/URSA 侧手工魔改并 vendored 进仓库。

## 3. 为什么确定是这个 commit

我不是只看文件名或目录结构，而是做了两层比对：

- 第一层：按 `path + blob sha1` 比对 `8eeb664:vllm` 和 upstream 历史提交。
- 第二层：优先排查 upstream `main` 历史，再对比临近 tag。

对最优候选 `098f94de...` 的结果是：

- 本地 vendored 树总文件数：`1348`
- upstream 该提交总文件数：`1342`
- 完全相同文件：`1330`
- 同路径但内容不同文件：`8`
- fork 新增文件：`10`
- upstream 存在但 fork 删除文件：`4`

等价成比例是：

- 以本地 vendored 树为基准，`1330 / 1348 = 98.66%` 文件完全一致
- 以共享路径为基准，`1330 / 1338 = 99.40%` 共享文件完全一致

这已经不是“风格接近”，而是标准的“在某个 upstream 基线之上做了有限文件级改造”的模式。

## 4. 它在 upstream 版本线里的位置

这个位置关系需要说清楚，因为这决定了它到底更像哪个版本。

- `098f94de...` 往回最近的前序 tag 是 `v0.6.3.post1`
- `098f94de...` 之后，upstream 还继续往前推进，到 `v0.6.4` 才打 tag

我实际统计得到：

- `098f94de...` 相对 `v0.6.3.post1`，位于其后 **265 个 main 提交**
- `098f94de...` 相对 `v0.6.4`，位于其前 **143 个 main 提交**

所以这份 fork 的准确版本定位不是：

- “大概 v0.6.4”

而是：

- “`v0.6.3.post1` 之后、`v0.6.4` 之前、接近 `2024-11-06 main` 的一份 fork 基线”

## 5. 为什么不是别的 tag

我对临近 tag 做了同一套相似度比对。关键结果如下：

| 候选 | 提交 | same | changed | local_only | upstream_only |
| --- | --- | ---: | ---: | ---: | ---: |
| 选中的 main 基线 | `098f94de...` | 1330 | 8 | 10 | 4 |
| `v0.6.4` | `02dbf30e...` | 982 | 341 | 25 | 82 |
| `v0.6.4.post1` | `a6221a14...` | 979 | 344 | 25 | 82 |
| `v0.6.5` | `2d1b9baa...` | 717 | 584 | 47 | 231 |
| `v0.6.6` | `f49777ba...` | 662 | 587 | 99 | 315 |
| `v0.6.6.post1` | `2339d59f...` | 659 | 590 | 99 | 317 |
| `v0.7.0` | `5204ff5c...` | 532 | 615 | 201 | 506 |

这个结果很明确：

- 它离 `v0.6.4` 虽然算近，但仍然明显不如 `098f94de...` 近
- 它离 `v0.6.5` / `v0.6.6` / `v0.7.0` 反而越来越远

也就是说，这份 fork 不是从更晚的 release tag 截出来的，反而是从更早的 `main` 分叉出去后，没有再系统性 rebase 到后续 release。

## 6. 真实 diff 结构

相对 `098f94de...`，完整 diff 统计是：

- `22 files changed`
- `5754 insertions(+)`
- `230 deletions(-)`

完整 patch 在：

- [vllm_8eeb664_vs_upstream_098f94de.diff](./vllm_8eeb664_vs_upstream_098f94de.diff)

统计摘要在：

- [vllm_8eeb664_vs_upstream_098f94de.stat](./vllm_8eeb664_vs_upstream_098f94de.stat)

文件分类清单在：

- [vllm_8eeb664_file_classification.txt](./vllm_8eeb664_file_classification.txt)

候选排名表在：

- [vllm_8eeb664_candidate_ranking.md](./vllm_8eeb664_candidate_ranking.md)

## 7. 改了哪些地方

### 7.1 修改了 upstream 原文件的部分

真正修改 upstream 原文件的只有 8 个：

- `.readthedocs.yaml`
- `vllm/engine/llm_engine.py`
- `vllm/entrypoints/chat_utils.py`
- `vllm/inputs/registry.py`
- `vllm/model_executor/models/registry.py`
- `vllm/transformers_utils/config.py`
- `vllm/transformers_utils/configs/__init__.py`
- `vllm/worker/model_runner.py`

其中真正有“功能意义”的核心改动主要是：

- `vllm/model_executor/models/registry.py`
  - 增加 `UrsaForConditionalGeneration` 注册项
- `vllm/transformers_utils/config.py`
  - 增加 `ursa -> UrsaConfig` 的 config registry 映射
- `vllm/transformers_utils/configs/__init__.py`
  - 导出 `DeepSeekMultiModalityConfig` 和 `UrsaConfig`
- `vllm/entrypoints/chat_utils.py`
  - 为 `deepseekqwen2vlm` 增加 `<|image|>` placeholder

其余几个文件更多是轻量改动：

- `.readthedocs.yaml`
  - Python 文档构建版本从 `3.12` 改成 `3.9`
- `vllm/worker/model_runner.py`
  - 增加了几行被注释掉的 CUDA graph 调试代码
- `vllm/engine/llm_engine.py`
  - 只有空行差异
- `vllm/inputs/registry.py`
  - 只有空行差异

### 7.2 fork 新增的文件

fork 新增了 10 个文件，分两类：

第一类是依赖兼容层：

- `attrdict/__init__.py`
- `attrdict/default.py`
- `attrdict/dictionary.py`
- `attrdict/mapping.py`
- `attrdict/merge.py`
- `attrdict/mixins.py`

第二类是模型/配置接入：

- `vllm/model_executor/models/deepseek_vl.py`
- `vllm/model_executor/models/ursa.py`
- `vllm/transformers_utils/configs/deepseek_vl.py`
- `vllm/transformers_utils/configs/ursa.py`

这说明这份 fork 的主要目标不是重做 vLLM core，而是往一个旧的 upstream 基线上强行塞入自定义多模态模型支持。

### 7.3 upstream 有、fork 删掉的文件

只有 4 个：

- `vllm/model_executor/layers/fused_moe/configs/README`
- `vllm/v1/tokenizer/__init__.py`
- `vllm/v1/tokenizer/detokenizer.py`
- `vllm/vllm_flash_attn/.gitkeep`

这里没有看到“大规模删功能”的迹象，更像是 vendoring 时没有把这些文件带进来，或者 fork 维护时顺手裁掉了非关键文件。

## 8. `URSA` 这一块是怎么嫁接进去的

这一点也比较清楚。

`vllm/model_executor/models/ursa.py` 文件头直接写了：

- 该文件包含 “Bytedance's Modifications”
- 同时保留了 `DeepSeek` 的版权声明

我又把 `ursa.py` 和同目录下新增的 `deepseek_vl.py` 做了直接 diff，结论是：

- `ursa.py` 不是从零写的
- 它明显是以 `deepseek_vl.py` 这一套多模态实现为模板改出来的
- 再结合 `attrdict` vendoring，可以判断这是一种“拿 DeepSeek 风格多模态实现做底座，改成 URSA 自己模型结构”的集成方式

所以这份 fork 的实际形态更像：

1. 先从 upstream `vllm` 的某个 `main` 历史点拉一份
2. 再把 `DeepSeek-VL` 支持塞进去
3. 再在其基础上派生 `URSA`
4. 最后补少量 core wiring，让 config / registry / chat placeholder 能跑通

## 9. 最终判断

如果要用一句话概括：

> `8eeb664...` 里的 `vllm/` 是一份以 upstream `vllm-project/vllm@098f94de...` 为最可能母本、位于 `v0.6.3.post1` 与 `v0.6.4` 之间的 pre-`v0.6.4` main fork，并在此基础上由 Bytedance 加入了 `DeepSeek-VL` / `URSA` 多模态模型支持与少量接线改动形成的魔改版。

这也是为什么你会直觉上觉得“像是直接复制了整个 vllm 仓库”：

- 绝大多数文件确实原样保留
- 真正魔改的点集中在极少数 registry / config / multimodal / model 文件
- 这是一个典型的 vendor-fork 结构，不是轻量 patch overlay 结构

