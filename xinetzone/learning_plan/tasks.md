# TVM 学习计划任务清单

## 阶段一：基础理论与入门（第1-2周）

### 第1周任务：ML编译基础与TVM概览

#### 任务1.1：理论学习
- [ ] 阅读 ML 编译器概述资料
- [ ] 了解 TVM 与其他 ML 编译器的比较
- [ ] 阅读 `docs/get_started/overview.rst`
- [ ] 理解 TVM 的设计理念和关键特性

#### 任务1.2：架构理解
- [ ] 学习 TVM 整体架构设计
- [ ] 了解核心组件介绍
- [ ] 理解编译流程概览

### 第2周任务：环境搭建与入门实践

#### 任务1.3：环境搭建
- [ ] 阅读 `docs/install/from_source.rst`
- [ ] 准备编译依赖环境
- [ ] 从源码编译 TVM
- [ ] 配置开发环境
- [ ] 验证安装成功

#### 任务1.4：快速入门
- [ ] 阅读 `docs/get_started/tutorials/quick_start.py`
- [ ] 运行快速入门示例
- [ ] 理解每一步的作用
- [ ] 修改示例代码进行测试

#### 阶段一练习
- [ ] **练习1**：TVM 环境搭建验证
- [ ] **练习2**：MLP 模型编译入门

---

## 阶段二：核心模块学习（第3-6周）

### 第3周任务：TensorIR基础

#### 任务2.1：TensorIR抽象概念
- [ ] 阅读 `docs/deep_dive/tensor_ir/abstraction.rst`
- [ ] 学习函数参数和缓冲区概念
- [ ] 理解循环迭代结构
- [ ] 掌握计算块（Block）概念

#### 任务2.2：块轴属性学习
- [ ] 学习块轴属性类型
- [ ] 理解空间轴和归约轴
- [ ] 掌握块轴绑定语法

### 第4周任务：TensorIR实践

#### 任务2.3：TensorIR编程
- [ ] 阅读 `docs/deep_dive/tensor_ir/learning.rst`
- [ ] 运行 `docs/deep_dive/tensor_ir/tutorials/tir_creation.py`
- [ ] 学习 TVMScript 语法
- [ ] 编写简单 TensorIR 程序

#### 任务2.4：TensorIR调度
- [ ] 运行 `docs/deep_dive/tensor_ir/tutorials/tir_transformation.py`
- [ ] 学习调度原语
- [ ] 练习手动调度优化
- [ ] 阅读相关测试用例

#### 阶段二前半练习
- [ ] **练习1**：TensorIR 矩阵乘法实现

### 第5周任务：Relax基础

#### 任务2.5：Relax抽象概念
- [ ] 阅读 `docs/deep_dive/relax/abstraction.rst`
- [ ] 学习计算图表示
- [ ] 理解数据流图概念
- [ ] 掌握函数和模块结构

#### 任务2.6：Relax编程
- [ ] 阅读 `docs/deep_dive/relax/learning.rst`
- [ ] 运行 `docs/deep_dive/relax/tutorials/relax_creation.py`
- [ ] 学习 BlockBuilder API
- [ ] 学习 TVMScript Relax 语法

### 第6周任务：Relax变换

#### 任务2.7：Relax变换系统
- [ ] 运行 `docs/deep_dive/relax/tutorials/relax_transformation.py`
- [ ] 学习 Pass 基础设施
- [ ] 阅读 `docs/arch/pass_infra.rst`
- [ ] 学习常用变换 Pass

#### 任务2.8：Relax实践
- [ ] 阅读 `python/tvm/relax/` 模块代码
- [ ] 学习测试用例最佳实践
- [ ] 编写 Relax 变换代码

#### 阶段二后半练习
- [ ] **练习2**：Relax CNN 模型构建

---

## 阶段三：高级特性与优化（第7-10周）

### 第7周任务：MetaSchedule基础

#### 任务3.1：MetaSchedule概述
- [ ] 学习 MetaSchedule 设计理念
- [ ] 了解核心组件架构
- [ ] 理解调优工作流程

#### 任务3.2：搜索空间定义
- [ ] 学习 Schedule Rule 概念
- [ ] 学习 Space Generator
- [ ] 学习搜索策略类型

### 第8周任务：MetaSchedule实践

#### 任务3.3：本地调优
- [ ] 阅读 `python/tvm/s_tir/meta_schedule/` 模块
- [ ] 学习本地调优配置
- [ ] 运行调优示例
- [ ] 分析调优结果

#### 任务3.4：RPC调优
- [ ] 学习 RPC 调优架构
- [ ] 配置远程调优环境
- [ ] 执行远程调优任务
- [ ] 管理调优数据库

#### 阶段三前半练习
- [ ] **练习1**：MetaSchedule 矩阵乘法调优

### 第9周任务：多目标代码生成

#### 任务3.5：Target抽象
- [ ] 学习 Target 抽象概念
- [ ] 阅读 `python/tvm/target/` 模块
- [ ] 了解不同硬件目标配置

#### 任务3.6：代码生成学习
- [ ] 学习 CPU 代码生成
- [ ] 学习 GPU 代码生成
- [ ] 了解特定硬件支持
- [ ] 阅读 `src/target/` 源码

### 第10周任务：自定义算子

#### 任务3.7：Topi算子库
- [ ] 学习算子库结构
- [ ] 阅读 `python/tvm/topi/` 模块
- [ ] 学习常用算子实现
- [ ] 练习使用算子库

#### 任务3.8：自定义算子开发
- [ ] 学习自定义算子开发流程
- [ ] 了解注册机制
- [ ] 实现简单自定义算子
- [ ] 集成到 TVM 中

#### 阶段三后半练习
- [ ] **练习2**：多目标编译实践

---

## 阶段四：实战应用（第11-16周）

### 第11周任务：模型导入基础

#### 任务4.1：前端架构学习
- [ ] 学习前端架构设计
- [ ] 了解支持的框架列表
- [ ] 理解导入流程

#### 任务4.2：PyTorch模型导入
- [ ] 学习 FX 导入方式
- [ ] 学习 Dynamo 导入方式
- [ ] 学习导出程序导入
- [ ] 阅读 `python/tvm/relax/frontend/torch/` 代码

### 第12周任务：模型导入实践

#### 任务4.3：ONNX模型导入
- [ ] 学习 ONNX 前端
- [ ] 练习模型转换
- [ ] 处理兼容性问题
- [ ] 阅读 `python/tvm/relax/frontend/onnx/` 代码

#### 任务4.4：其他框架导入
- [ ] 学习 TensorFlow Lite 导入
- [ ] 学习 StableHLO 导入
- [ ] 对比不同导入方式

### 第13周任务：端到端优化

#### 任务4.5：优化管线配置
- [ ] 学习内置管线
- [ ] 学习自定义管线
- [ ] 阅读 `docs/how_to/tutorials/e2e_opt_model.py`
- [ ] 运行端到端优化示例

#### 任务4.6：LLM优化学习
- [ ] 阅读 `docs/how_to/tutorials/optimize_llm.py`
- [ ] 学习大模型优化策略
- [ ] 了解 KV Cache 优化

### 第14周任务：部署实践

#### 任务4.7：跨编译与RPC
- [ ] 阅读 `docs/how_to/tutorials/cross_compilation_and_rpc.py`
- [ ] 学习 RPC 系统架构
- [ ] 练习远程执行
- [ ] 学习跨平台调试

#### 任务4.8：模型导出与加载
- [ ] 阅读 `docs/how_to/tutorials/export_and_load_executable.py`
- [ ] 学习可执行模块导出
- [ ] 学习模型序列化
- [ ] 练习运行时加载

#### 阶段四前半练习
- [ ] **练习1**：ResNet 模型端到端优化
- [ ] **练习2**：模型部署实践

### 第15周任务：综合项目（前半）

#### 任务4.9：项目启动
- [ ] 选择预训练模型
- [ ] 设计项目架构
- [ ] 制定实施计划

#### 任务4.10：模型导入与验证
- [ ] 导入选择的模型
- [ ] 验证模型正确性
- [ ] 建立基准性能

### 第16周任务：综合项目（后半）

#### 任务4.11：模型优化
- [ ] 执行图级优化
- [ ] 执行张量级调优
- [ ] 针对目标硬件优化
- [ ] 性能对比分析

#### 任务4.12：项目收尾
- [ ] 导出优化模块
- [ ] 实现推理接口
- [ ] 编写部署文档
- [ ] 撰写项目报告
- [ ] 准备项目答辩

---

## 综合项目清单

### 项目准备
- [ ] 确定项目主题
- [ ] 选择目标模型
- [ ] 确定目标硬件平台
- [ ] 制定项目计划

### 项目实施
- [ ] 完成模型导入
- [ ] 完成模型优化
- [ ] 完成性能测试
- [ ] 完成部署实现

### 项目交付
- [ ] 提交项目代码
- [ ] 提交性能报告
- [ ] 提交部署文档
- [ ] 提交项目总结
