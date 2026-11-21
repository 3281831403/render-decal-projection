# render-decal-projection

## 题面

给墙面加投影贴花后，斜面上出现条纹、边缘还渗色：投影基向量没有正交化，贴花被斜切；法线为零时没有报错，矩阵退化；裁剪用包围球判断，盒外片元漏裁；深度偏移沿视线而不是法线，斜面渗色；边缘淡出不钳制，alpha 为负后反转；混合重复乘 alpha，边缘发暗；排序用不稳定 sort，同距离闪烁；尺寸超限不报错，纹理拉伸；批量共用矩阵，贴花之间串；size=0 不校验，除零。decal/project.py 是核心，tests/test_project.py 15个红。要修的是：正交基、零法线校验、立方体裁剪、法线深度偏移、边缘 alpha 钳制、单次 alpha 混合、稳定排序、尺寸校验、逐贴花独立矩阵与 size=0 拒绝。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态应恰好为 `15 failed`（无 collection/import/syntax error，15 个失败均来自 8 个 `NotImplementedError` 桩）。测试离线运行，无网络、无随机性，仅需 Python 3 与 pytest。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 数据约定

- 贴花：`{"matrix": [16 个 float，4×4 行主序], "size": float}`。
- 投射体 / 片元：`{"pos": [x, y, z], "normal": [x, y, z]}`，可带额外字段。
- 贴花局部空间是单位立方体 `[-0.5, 0.5]³`；所有几何比较容差为 `1e-6`。

## 约束

`tests/test_project.py` 是评测契约，不可修改。实现时只修改 `decal/project.py`，仅使用标准库（`math`）与 pytest，不得引入网络或随机行为。
