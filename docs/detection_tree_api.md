# 检测树：计算能力与单项计算接口

当前支持节点能力查询和单项计算。每次请求只执行所需依赖，在请求内复用结果。
支持固定顺序的单图计算计划、测试模式判断树及运行记录保存；尚无跨请求统一缓存或批次任务。OCR 模型按需加载。
后续执行器、前端树应共用这里的 operation ID，不使用界面文字或文件名作为标识。

## 接口

### 当前实现范围

本版本是人工演示规则驱动的检测流程树，按有序深度优先遍历执行，命中后停止。
尚未训练分类决策树，也未实现类别优先级或计算成本驱动的自动树优化。

### 删除草稿节点

`POST /api/detection/trees/nodes/delete` 接收 `{tree, node_id, mode}`。
`mode=subtree` 删除节点及全部子孙；`mode=promote_children` 只删除当前节点，
子节点按原顺序接回原父节点，在被删除节点原位置插入。根节点不可删除。
成功返回完整草稿 tree、removed_node_ids、parent_id、status=draft 和 requires_validation=true。
删除不自动保存，不修改其他计算或判断规则；允许依赖暂时缺失，确认完成后统一校验。
非法删除返回 HTTP 400，error.code 为 invalid_node_deletion。

### 保存和读取用户树配置

| 方法与路径 | 功能 |
|---|---|
| POST /api/detection/tree-configs | 新建配置，返回 201 |
| GET /api/detection/tree-configs | 获取摘要列表 items |
| GET /api/detection/tree-configs/{id} | 读取完整配置 |
| PUT /api/detection/tree-configs/{id} | 更新配置，返回 200 |

新建请求包含 name、status、tree。status 为 draft 或 published；更新还必须提供
expected_version，等于上次读取的配置 version。版本过期返回 409 version_conflict，
并返回 current_version；不存在的配置返回 404。
草稿只校验结构和节点定义，published 必须通过完整依赖校验。
返回记录包含 id、name、version、status、tree、created_at、updated_at、published_tree、published_version。
tree 是最近保存的编辑版本；published_tree 是最后通过正式校验保存的树，首次仅保存草稿时为空。
后续保存草稿不会覆盖 published_tree。published 不代表切换某个全局执行策略：
前端选定配置后仍需将要执行的完整树提交 /tree-runs，执行端会再次校验。
配置默认存入 data/detection_trees/trees.sqlite3，可用 DETECTION_TREE_STORE 设置存储目录。
SQLite 事务保护版本检查与写入，避免多个页面或进程同时保存造成覆盖。
存储的是最新编辑版本及最后正式版本，不提供全部历史版本恢复，也尚无按用户隔离的权限管理。

### 增加草稿节点

`POST /api/detection/trees/nodes` 接收完整草稿树并返回新草稿，不执行计算、不持久化、不修改当前生效树。
请求字段为 `tree`、`parent_id`、`node` 和可选 `index`（从 0 开始，省略或 null 表示追加到末尾）。
例如在 OCR 下的第一个位置新增文字面积计算：

```json
{
  "tree": {"schema_version": 1, "root_id": "root", "nodes": [
    {"id": "root", "type": "group", "children": ["ocr"]},
    {"id": "ocr", "type": "compute", "operation": "ocr.structured", "children": []}
  ]},
  "parent_id": "ocr",
  "index": 0,
  "node": {"type": "compute", "operation": "text.geometry", "label": "文字面积"}
}
```

演示判断节点使用同一接口，node 示例：

```json
{
  "type": "pattern", "demo": true, "label": "测试模式D",
  "rule": {"combine": "all", "conditions": [
    {"operation": "text.geometry", "field": "text_area_ratio", "operator": "gt", "value": 0.3}
  ]}
}
```

node.id 可自行提供唯一值，省略则后端生成；新增节点必须没有子节点。
父节点只允许计算节点或分组节点，判断节点仍为叶子。
成功 HTTP 200 返回 `{schema_version:1, status:"draft", tree:完整新树, node_id, parent_id, index, requires_validation:true}`。
已有节点及其子树不变，只在指定父节点的 children 中插入新 ID。重复使用同一 operation 的节点允许存在，
执行时沿用本轮结果复用机制。未知计算、重复 ID、非法规则、无效父节点、越界位置等返回 HTTP 400、
`error.code=invalid_node_addition`。
新增时检查结构及节点配置，但不检查计算依赖先后，允许继续编辑未完成的草稿。
用户确认完成后仍须调用 `/trees/validate`；正式执行 `/tree-runs` 也始终执行完整依赖校验。

### 树调整的依赖约束

后端能力目录中的 `dependencies` 是计算节点的前置约束；判断节点的前置约束来自
`rule.conditions[].operation`。不要求它们是树上的直接父子，只要求在实际深度优先遍历中先完成。
移动整个子树后必须重新校验完整树，不能只比较两个节点。后端不会自动重排。

前端拖动后，将候选树提交 `POST /api/detection/trees/validate`，成功才应用修改。
依赖不合法时返回 HTTP 400，例如：

```json
{
  "valid": false,
  "error": {
    "code": "invalid_dependency_order",
    "message": "Tree dependencies must be computed before their consumers",
    "execution_order": ["root", "rules", "ocr"],
    "issues": [{
      "node_id": "rules",
      "operation_id": "text.rules",
      "dependency_operation_id": "ocr.structured",
      "producer_node_ids": ["ocr"],
      "reason": "not_computed_before",
      "message": "节点 rules 需要先完成 ocr.structured；该计算没有排在此节点之前。"
    }]
  }
}
```

`issues` 一次列出全部直接依赖问题；前端可根据节点 ID 高亮。`missing_computation` 表示
树及补充计算中均未配置该计算（producer_node_ids 为空）；`not_computed_before` 表示已配置，
但没有出现在使用者之前。判断节点的 operation_id 为 null，其 dependency_operation_id 指向规则输入。
补充计算用 `completion:索引` 标识，执行顺序在主树之后；execution_order 仅列主树节点。
树结构或规则本身无效时优先返回原有 invalid_tree 错误，不分析依赖顺序。
`POST /api/detection/tree-runs` 执行前同样校验，依赖失败时返回相同 error 对象和 HTTP 400，
不会识别图片或启动计算。运行中的计算失败仍按 failed/blocked 记录，不当作未匹配。

- `GET /api/detection/capabilities`：返回完整清单。
- `GET /api/detection/capabilities/{operation_id}`：返回单项说明；不存在时返回 404 和 `error.code=operation_not_found`。

服务由现有 Flask app 注册，默认端口 5000。Vite 已配置 `/api/detection` 专用代理，
前端可直接 fetch 相对地址；生产部署也需要将此前缀路由至 Flask，不能转发给模型服务。

## 单项计算（第二步）

`POST /api/detection/compute`，multipart/form-data：

- `file`：图片文件。
- `operation_id`：清单中的计算 ID。
- `ocr_input`：可选；不传时自动调用本地 DeepSeek-OCR，传入时优先使用该结果。
- `format`：grounding（默认）或 tsv。

图像梯度和灰度不要求 OCR 输入。OCR 输入必须对应上传图片的 EXIF 校正后尺寸。
默认 Flask 应用启用 OCR 时，会配置 `DETECTION_OCR_PROVIDER`，复用已有 DeepSeek-OCR 服务。
执行到 OCR 节点时才识别校正后的 PIL RGB 图片，使用 markdown grounding 模式取得文字及文本框。
临时图片和推理输出在调用结束（含失败）后清理，本轮结果由执行器保存供后续节点复用。
`format` 指定手动输入格式；默认自动识别始终使用 grounding。自定义回调可通过
`output_format` 属性声明返回格式，否则沿用请求 format。
`--no-ocr` 禁用自动识别；此时仍可传入 ocr_input。识别失败或缺少定位框不会伪装成零特征：
单项接口返回错误，树执行记录 failed/blocked，并继续独立分支。
这同样适用于 `/compute-sequence` 和 `/tree-runs`，无需另行调用 OCR 接口。

成功返回 `values`、实际执行的 `computations`（含独立计算耗时）以及 `cache_scope=request`。
数组暂以 JSON 内联返回（灰度二维数组、windows 对象），仅适合单图调试；持久化结果阶段改为引用。
缺少文件或未知计算为 400，输入无效为 422，内部计算失败为 500。

Python 调用方可在同一个 ComputationSession 中多次 compute，不会重复执行已完成依赖。
这不是跨请求缓存，刷新请求后会重新计算。

## 固定顺序与结果共享（第三步）

`POST /api/detection/plans/validate` 接收 JSON：

```json
{"operations":["ocr.structured","text.rules","image.gray","image.gradient","text.gradient_overlap"]}
```

返回 `valid=true` 和原顺序。每项依赖必须在前面明确列出，否则返回 400、
`error.code=invalid_order` 和原因；不自动改变用户顺序。灰度等内部准备步骤也要列入计划。
列表最多 1000 项。此接口不保存策略，前端调整后可先请求它检查。

`POST /api/detection/compute-sequence` 接收 multipart/form-data：

- `file`、`ocr_input`、`format` 同单项计算接口。
- `operations`：上例数组的 JSON 字符串。

先完整检查顺序，再进行计算。每项直接按 operation ID 读取本次结果表，
不会递归补算依赖。重复计算名称只执行第一次，后续记录 `reused=true`。
目前参数固定，因此相同 operation ID 对应同一项计算；后续允许不同参数时需扩展结果标识。

响应包含：

- `status`：全部完成为 completed，有失败或受阻为 incomplete。
- `results`：按 operation ID 保存的成功结果，只返回一份，不包含失败的零值替代品。
- `computations`：每项的 sequence、operation_id、reads、status、reused、elapsed_ms。
- `blocked_by`：受阻时列出未成功完成的直接依赖。
- `cache_scope=request`：每次 HTTP 请求使用新的结果表，结束后不持久化。

合法请求的执行记录使用 HTTP 200 返回，前端必须检查顶层及节点 status。
计划或文件缺失返回 400，图片解码或格式错误返回 422。
计算失败记为 failed，不自动重试；依赖它的节点记为 blocked，独立分支继续执行。
重复失败项仍保持失败；耗时仅为当前计算，不包含前置计算时间。

这一步执行的是已有计算能力的固定列表，不包含模式匹配、提前结束、树的编辑或分支跳过。
原 `/compute` 仍保留按需补齐依赖的行为，新固定顺序接口不会静默补算。

## 返回契约

## 树执行与运行记录（第四、五步）

| 接口 | 用途 |
|---|---|
| GET /api/detection/trees/demo | 获取初始测试树，可修改 children 顺序与测试阈值 |
| POST /api/detection/trees/validate | JSON 请求体为完整树，返回 execution_order；非法结构为 400 |
| POST /api/detection/tree-runs | multipart：file、tree（JSON 字符串）、ocr_input、format；同步执行并保存，返回 201 |
| GET /api/detection/tree-runs/{run_id} | 按运行 ID 读取已保存的完整记录；不存在或非法 ID 返回 404 |

树结构包含 schema_version=1、root_id、nodes、可选 completion_operations。
节点包含 id、type（group/compute/pattern）、children；compute 还包含 operation。
children 顺序决定有序深度优先遍历。不同节点可引用相同 operation，结果只算一次，
但仍访问后续子节点。父节点计算失败也会继续遍历，相关后续节点按依赖标为 blocked。
树中每个节点只能有一个父节点，禁止循环、孤立节点和重复 ID；completion: 前缀为系统保留。

pattern 必须是叶节点且 demo=true。rule.combine 为 all 或 any，conditions 为非空列表，
每项包含 operation、field、operator（gt/gte/lt/lte/eq）、value（有限数字）。
仅允许已登记的数值字段和显式比较，不执行任意表达式。缺失或空值记为未知，
all 中明确失败条件可判未匹配，any 中明确成功条件可判匹配，其他不充分情况记为 blocked。

所有前置计算必须已安排在遍历顺序中；校验在计算之前完成，不静默重排。
首次测试模式完整匹配后，剩余节点记录 skipped；匹配仅表示满足测试条件，不能确认攻击。
全部未匹配时按 completion_operations 补齐声明的特征集合；当前演示仅覆盖七项能力，
不代表项目全部通用特征已提取。没有匹配且存在失败/未知时，返回 incomplete。

返回并保存 run_id、sample_id（图片字节哈希）、完整树快照、起止时间、records、events、results、
matched_pattern、stop_reason、has_errors、completion_satisfied。
records 保存节点顺序、独立耗时、依赖 reads、复用标记及规则实际值和阈值。
group 的 visited 表示已展开，并非整棵子树均成功。
events 保存 running 与结束状态，供前端回放；当前接口同步返回，不是实时 SSE。
如前面发生错误、后续仍匹配，status=matched 且 has_errors=true，保留所有失败证据。

记录默认原子写入 data/detection_runs/{run_id}.json；可用服务端 DETECTION_RUN_STORE 配置位置。
数组当前内联保存，图片原文件未保存，仅保存其哈希；前端仍需保留原图关联。
记录供查询与比较，不自动用于下一次运行缓存。仅保存已结束任务，尚不支持运行中断恢复。

## 能力字段

顶层 `schema_version` 标识接口结构版本，`catalog_version` 标识清单版本。
`operations` 中每项包括：

| 字段 | 含义 |
|---|---|
| id / version | 稳定计算标识、计算契约版本 |
| label / kind | 显示名称、节点种类（当前均为 compute） |
| dependencies | 必须取得结果的计算 ID；这是计算依赖，不是树的父子关系或遍历顺序 |
| inputs | 无依赖时使用 sample.image，否则引用对应依赖的输出包 |
| outputs | 输出字段名、类型和是否允许空值 |
| parameters | 当前为空；尚未开放可调参数 |
| implementation | 现有代码位置、接入状态及待改造说明，仅供开发参考 |
| executable | 当前均为 true，表示已提供单项计算适配器；OCR 仍需要输入或服务端提供者 |

前端只有在计算接口成功返回后才能显示完成。
`artifact` 类型预留给大型数组或区域结果，当前计算接口使用内联数据。
OCR boxes 包含 box、text、confidence；块级结果不是词级精确定位。
数值为 null 表示缺少可计算依据，不代表零或模式未匹配。

## 尚未实现的部分

前端树编辑、实时进度推送、跨请求共享缓存，以及训练或成本优化生成的分类决策树尚未实现。
cache_identity_fields 是未来跨请求缓存的预留约定；当前已有单次执行内的结果复用。

能力清单的 `patterns` 当前为空，测试模式通过 `/trees/demo` 的树定义提供，明确标注为演示规则；
目前这七项能力也不是完整的通用特征集合，不能宣称已支持全部特征补齐。

## 验证

在项目根目录运行：

```powershell
$env:PYTHONPATH='backend'
& '.\.venv-ocr\Scripts\python.exe' -m unittest discover -s backend/tests -p 'test_detection*.py' -v
```
