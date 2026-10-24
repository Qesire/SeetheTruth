/* Shared PC + Android teaching curriculum. A single source for link targets,
 * explanations and falsifiable checkpoint questions. No model output exports. */
(function(root,factory){"use strict";const a=factory();if(typeof module==="object"&&module.exports)module.exports=a;root.SeetheTruthCurriculum=a;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const lessons=Object.freeze([
  {id:"building",group:"01 · 模型积木",title:"把大模型拆成可执行积木",href:"learn.html",
   level:"基础 → 进阶",theory:"先重建张量形状、连接约束与运算顺序，再解释 Attention、FFN、归一化和梯度。先预测每一步输出形状，再运行微型数值例子。",
   why:"真实模型往往用多篇论文中的机制拼装；只记网络结构名称，不能推导数据流或复杂度。",
   check:"不同积木的端口维度匹配，是否足以保证完整模型数学语义相同？",
   answer:"不够。维度匹配只是必要条件，还要检查 Mask、状态、时间索引、归一化与残差作用点。"},
  {id:"papers",group:"02 · 论文依据",title:"从论文理解‘为什么这样设计’",href:"paper_lab.html",
   level:"基础 → 研究",theory:"比较论文提出的原始算子与重写后的计算图，追踪优化的是激活规模、通信、梯度路径还是建模能力。",
   why:"有些方法改变数值语义，有些仅改变等价计算顺序；不区分就无法进行严谨消融。",
   check:"一次 Kernel Fusion 一定改变模型的数学函数吗？",answer:"不一定。合法的融合可以保持数学函数，但数值舍入、精度与设备实现可能发生变化。"},
  {id:"multimodal",group:"03 · 多模态",title:"图像、音频和视频 Token 计算流",href:"learn.html",
   level:"进阶",theory:"图像 Patch/Video Tubelet/音频谱图先映射到统一或对齐的 Token 表示，再进入 Attention 与跨模态计算。比较各自的时空或时频位置依赖。",
   why:"模态转换不仅是通道数变化，还可能引入重采样、时间对齐、非均匀序列长度与专用 Mask。",
   check:"视频 Tubelet 和逐帧图像 Patch 会自动拥有相同的时间信息吗？",
   answer:"不会。Tubelet 会直接编码局部时间块；逐帧 Patch 需要额外的位置编码或时间混合机制。"},
  {id:"loop",group:"04 · 前沿架构",title:"GQA / MoE / Loop Transformer 的成本权衡",href:"tradeoffs.html",
   level:"进阶 → 研究",theory:"把参数量、KV Cache、单 Token 激活专家数、共享参数重复使用次数分别列式，区分参数存储与实际计算。",
   why:"更少参数或更少激活并不自动产生更低延迟，访存、调度、Token 分布和通信也会决定性能。",
   check:"Loop Transformer 将共享层调用 N 次，参数量和计算量都变成原来的 N 倍吗？",
   answer:"不是。参数可共享而保持接近不变，但重复执行通常会提高计算量；状态与停止条件还会影响实际耗时。"},
  {id:"cache",group:"05 · 推理基础",title:"KV Cache：位置、复用和分页",href:"cache.html",
   level:"进阶",theory:"观察注意力的 Query 与历史 Key/Value 读写，区分序列长度、逻辑位置和物理 Block Table 引用。",
   why:"Decode 的主要压力未必是矩阵乘法峰值，KV 的容量、带宽与碎片会限制批量大小和吞吐。",
   check:"两个请求共享同一个物理 KV Block 时，结束其中一个请求可立即回收此块吗？",
   answer:"不能；只有所有引用者都释放后 RefCount 才归零并回到空闲队列。"},
  {id:"prefill",group:"06 · DeepInfer",title:"Prefill / Decode 与多模态 Chunk 边界",href:"deepinfer_prefill_lab.html",
   level:"研究",theory:"依据官方 C++ Scheduler：NORMAL 在周期 ALL/ANY 投票点统一 DP 内的步型；第二轮确认实际 Prefill 调度。Chunk 上限还要考虑 CP 对齐、Checkpoint 和图片 PAD 保护。",
   why:"ANY 为孤立请求提供尝试机会，ALL 减少无请求 Rank 的空 Prompt；第二次投票避免资源不足时陷入无产出的 Prompt 循环。",
   check:"如果 ANY 投票通过，但所有 Rank 都因 KV 内存不足没有排上 Prefill，应继续做空 Prompt 吗？",
   answer:"不应该。DeepInfer 会进行第二轮是否实际排出 Prefill 的投票，全部失败就回退 Decode。"},
  {id:"scheduler",group:"07 · DeepInfer",title:"Scheduler → KV → Worker 的执行证据",href:"deepinfer_scheduler_lab.html",
   level:"研究",theory:"观察 Decode 优先级、KV 资源抢占、WAITING 重新准入和 Worker 对应 Forward；每步都核对请求是否持有实际物理 Block ID。",
   why:"错误的调度可能让已经释放 KV 的请求继续 Forward。先验证所有权，再讨论性能。",
   check:"请求被 Preemption 释放 KV 后，能直接进入下一步 Decode 吗？",
   answer:"不能。若是重计算而非终止，必须先经过后续 Prefill 准入并重新建立可用资源。"},
  {id:"worker",group:"08 · 推理系统",title:"ZOS：跨 Step 事件和采样状态",href:"deepinfer_worker_lab.html",
   level:"研究",theory:"跟踪上一轮 Sample → 当前 Metadata Correction → Forward → Postprocess/Flush 的事件依赖，区分抽象 CPU/GPU 事件与真实设备 Trace。",
   why:"流水化试图重叠主机后处理与设备前向，但跨步依赖和状态一致性比单步 FLOPs 更重要。",
   check:"模拟时间线中的 CPU/GPU 阶段重叠，能直接证明线上加速吗？",
   answer:"不能。这里只证明给定假设耗时下的合法依赖与可能重叠；实测加速需要真实设备追踪。"},
  {id:"engineering_journey",group:"09 · 工程实践",title:"从数据到推理 · 十二阶段工程学习",href:"engineering_journey.html",
   level:"基础 → 进阶",theory:"通过真正的小词表 Bigram 模型逐步完成训练集/验证集隔离、标签移位、参数维度、交叉熵、解析梯度、AdamW、Checkpoint 与自回归推理。",
   why:"完整工程实践不只包含模型计算图，也包含随机性、数据谱系、优化器状态和可恢复的推理接口；这些因果链必须可重放。",
   check:"即使固定了初始化随机种子，如果中断训练时只保存模型参数，却丢失优化器状态和数据采样进度，可以完全恢复原来的训练轨迹吗？",
   answer:"不能。还需要优化器状态、数据采样顺序与游标、版本和待累积梯度等状态；具体内容取决于中断位置。"},
  {id:"engineering_pipeline",group:"10 · 工程实践",title:"进阶 · 六阶段训练流水线",href:"engineering_pipeline.html",
   level:"进阶 → 研究",theory:"执行 Epoch Shuffle、Microbatch 梯度积累、Clip、Warmup/Cosine、半累积状态 Checkpoint/Restore 和同一模型的推理续接。",
   why:"梯度累计必须按实际样本权重聚合；重放需要保持参数、优化器、采样游标、缓冲梯度与调度器步数一致。",
   check:"在未完成 Gradient Accumulation 时保存的 Checkpoint，只包含 W、m、v 和 Step 足够吗？",
   answer:"不够，还必须保存或可确定性重建累计梯度、有效样本数、Epoch/Cursor 与相关数据来源，才能精确续训。"}
 ]);
 const ids=new Set(),targets=new Set();
 for(const lesson of lessons){
  if(ids.has(lesson.id)||!/^[a-z0-9_-]+$/.test(lesson.id))throw Error("duplicate or invalid lesson id");
  if(!/^[a-z0-9_-]+\.html$/.test(lesson.href))throw Error("invalid teaching destination");
  if(!lesson.theory||!lesson.why||!lesson.answer)throw Error("missing teaching explanation");
  ids.add(lesson.id);targets.add(lesson.href);
 }
 return Object.freeze({VERSION:"2026.10.24",lessons,targets:Object.freeze([...targets]),lookup:id=>lessons.find(x=>x.id===id)||null});
});
