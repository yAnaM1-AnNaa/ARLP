# yolo

针对InternVL对图片中物体数量识别有误的情况，先用yolo将想要识别的物体切割出来，然后输入MHACoT。

将图片中的物体切割出来进行提问。

| 模型                                                         | 尺寸 (像素) | mAPval 50-95 | 速度 CPU ONNX (毫秒) | 速度 T4 TensorRT10 (毫秒) | 参数 (M) | FLOPs (B) |
| :----------------------------------------------------------- | :---------- | :----------- | :------------------- | :------------------------ | :------- | :-------- |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640         | 54.7         | 462.8 ± 6.7          | 11.3 ± 0.2                | 56.9     | 194.9     |

将yolo所有可以检测的物体列表提取

![image-20251122144822536](./assets/image-20251122144822536.png)

检测结果

![result](./assets/result.jpg)

修改prompt

question1--强调 "inside the bounding box"，并将 "interacts" 改为 "is designed to interact" (因为图中无人)

question3-从 "Describe... in the image" (描述图中现状) 改为 "Predict the most likely interaction" (预测潜在交互)



新的回答

![image-20251122175652402](./assets/image-20251122175652402.png)

读取yolo分割出来的部分（也就是明确告知object所在的区域），然后使用修改后的prompt。针对每个物体单独进行一次问答。

对于第一个回答效果不好，回答说headboard有adjustable massage feature，交互部位直接错了

第二个回答效果很好，交互部位是Mattress，几何特征是smooth, soft surface， 交互是lie down to sleep or rest。

对于additional affordance， Standing on the feet to push the mattress意义不明，需要筛选



更换参数和新的prompt

修改参数：generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=3)模型在每一步生成时，会同时保留**概率最高的 3 条路径**（候选句子）。等到句子生成结束时，再从这 3 条完整的路径中选出总概率最高的那一条。消除模型幻觉并保留逻辑能力，但是会增加计算量

prompt1：Look at the {label} inside the red bounding box within the panoramic scene. Analyze its spatial relationship with the surroundings.（这里利用了全景图的优势）

prompt2：Considering the spatial context you just analyzed, identify the standard, primary function of this {label}.  Does the visible geometric structure inside the box support this function?(e.g., for a bed, flat surface for sleeping; for a chair, seat and backrest for sitting). Avoid describing rare or hidden features like 'massage' unless clearly visible.

Spatial Context: 1. The bed is not blocked by other furniture.
2. The bed is placed in the center of the room.
3. The bed is not facing a specific direction such as a TV or window.
4. Core Analysis: Considering the spatial context analyzed, the standard, primary function of this bed is to provide a flat surface for sleeping. The visible geometric structure inside the box supports this function, as it is *a bed with a mattress and a striped bedspread*. The bed is placed *in the center of the room*, *not blocked by other furniture*, and is not facing a specific direction such as a TV or window, making *it suitable for sleeping.*

修改后*a bed with a mattress and a striped bedspread， suitable for sleeping*都相当准确。



Spatial Context: 

The bed is not blocked by other furniture.

2. The bed is placed in the center of the room.
3. The bed is not facing a specific direction such as a TV or window.
4. Core Analysis: Considering the spatial context analyzed, the standard, primary function of this bed is to provide a flat surface for sleeping. The visible geometric structure inside the box supports this function, as the bed has a flat surface suitable for sleeping. The bed is placed in the center of the room, not blocked by other furniture, and is not facing a specific direction such as a TV or window, which further supports its primary function as a sleeping area.



# uad

点云和fusion得到的部分照片如下

![image-20251124173935207](./assets/image-20251124173935207.png)

![image-20251124174441096](./assets/image-20251124174441096.png)

![image-20251124174222761](./assets/image-20251124174222761.png)

![image-20251124174252398](./assets/image-20251124174252398.png)

可以看到聚类得到的结果还是比较不错的，各部分比较清晰，点云质量也可以

原先的运行结果（更多的物体）

![affordance_map_sit-on_1](./assets/affordance_map_sit-on_1.png)

![affordance_map_sit-on_2](./assets/affordance_map_sit-on_2.png)

如果使用更长的text_query，那么就热力图区域会直接消失。（上面用的text_query是sit，换成the area to sit on）其中sit对应的是red区域

![image-20251125155657849](./assets/image-20251125155657849.png)

![affordance_map_sit-on_1](./assets/affordance_map_sit-on_1-1764080720502-7.png)

![Blue-affordance_map_the area to sit on](./assets/Blue-affordance_map_the area to sit on.png)

使用dinov2 vit large， 将sentence transformer更改为openai-text-embedding-3-large

原先配置为dinov2 vit small， all-MiniLM-L6-v2

![Blue-affordance_map_sit on](./assets/Blue-affordance_map_sit on.png)

![Blue-affordance_map_sit on2](./assets/Blue-affordance_map_sit on2-1764081767061-11.png)

![Blue-affordance_map_sit on-bench](./assets/Blue-affordance_map_sit on-bench.png)



# 网络架构

基于您提供的图像，这是对 `Conv2DFiLMNet` 网络架构及其关键组件的解释：

### 网络总体架构 (Conv2DFiLMNet)

`Conv2DFiLMNet` 是一个神经网络，设计用于同时处理图像和语言输入。

**输入**：

**Image Input**：一个形状为 `(B, 1024, H, W)` 的图像特征张量，其中 B 是批次大小 1。

**Language Embedding**：一个形状为 `(B, 1536)` 的语言特征向量 2。这代表了语言信息的语义嵌入。

**处理流程**：图像输入依次通过三个 `Conv2DFiLMBlock`。每个块都会处理图像特征，并将维数从 `(B, 1024, H, W)` 逐步转换为最终的 `(B, 1, H, W)` 输出 3。

**FiLM Conditioning**：关键在于，`Language Embedding` 被注入到每一个 `Conv2DFiLMBlock` 中，用于调节图像特征的处理过程 4。这种机制被称为 "FiLM Conditioning"（Feature-wise Linear Modulation，特征线性调制）。

### 关键组件详解

![image-20251126224019313](./assets/image-20251126224019313.png)

图像下方的 `Conv2DFiLMBlock` 详细图解（配置为 `film_place='after', mode='zero'`）展示了 FiLM 调节的具体工作方式：

1. **Language Embedding 的作用**：

   在每个 `Conv2DFiLMBlock` 内部，`Language Embedding`（图中的 "Input Lang Emb"）被送入一个名为 `FiLMBlockZero` 的模块 5。

   在 `FiLMBlockZero` 内部，语言嵌入通过两个并行的全连接层（`fc_gamma` 和 `fc_beta`）进行处理 6。

2. **Gamma (γ) 和 Beta (β)**：

   `fc_gamma` 层从语言嵌入中计算出调节参数 **gamma (γ)** 7。

   `fc_beta` 层从语言嵌入中计算出调节参数 **beta (β)** 8。

   这些参数是特定于语言输入的，用于对图像特征进行个性化的调节。

3. **FiLM Conditioning (特征线性调制)**：

   + 图像特征（图中的 "Input X"）首先经过卷积层 (`Conv2d Layer`) 和可选的归一化层 (`Norm Layer (opt.)`) 9。

   + 然后，这些中间特征 x 进入 FiLMBlockZero，并应用以下公式进行调节：

     $$x = (1 + \gamma)x + \beta$$

   + 在这个公式中：

     `x` 是来自图像路径的中间特征图 10。

     **(1 + γ)** 作为一个缩放因子（scale factor），根据语言信息增强或抑制特征 11。由于 `mode='zero'`，γ 初始化接近于0，使得初始调节接近于恒等映射 (1 * x)，网络可以从这里开始学习 12。

     **β** 作为一个平移因子（shift factor），对特征进行偏移 13。

   这种机制允许语言信息通过线性缩放和平移来“指导”网络如何处理图像特征，从而实现基于语言的图像理解或生成任务。调节后的特征随后通过一个可选的激活层 (`Activation (opt.)`) 产生最终的 `Output Features` 14。

   

   ![image-20251126224748194](./assets/image-20251126224748194.png)

基于您提供的 Python 代码和图像 1，以下是对流程图中特定模块的解释，以及对整个图表正确性的评估：





### 模块解释

1. **Natural Language Query (红色方框)**:

   + **对应代码**: 这是输入到系统的原始文本字符串。在 `inference.py` 的 `main` 函数中，有一个示例查询 `text_query = "contain"` 2。这个字符串随后被作为 `text` 参数传递给 `AffordanceInference.predict` 方法 3。
   + *含义**: 它代表了用户想要在图像中寻找的 affordance（功能可见性）的自然语言描述。

2. **Preprocessing & Transforms (第二个蓝色方框)**:

   **对应代码**: 在 `inference.py` 的 `predict` 方法中，原始图像 `img_np` 首先经过 `transform_imgs` 函数进行处理：`proc = transform_imgs(img_np, blur=False)[0]` 4。

   **含义**: 这一步涉及将输入的 RGB 图像转换为适合输入到冻结的 DINO 模型所需的格式，可能包括标准化、调整大小或其他必要的变换。

3. **Text Embedding Function (红色方框)**:

   **对应代码**: 在 `inference.py` 中，`AffordanceInference` 类在初始化时接收一个 `text_embedding_func` 5。在 `predict` 方法中，这个函数被用来将自然语言查询转换为向量：`lang_emb = torch.from_numpy(self.text_embedding_func(text))` 6。配置文件决定了具体使用哪种嵌入方法（例如，来自 OpenAI 的嵌入）7。

   **含义**: 这个模块负责将人类可读的文本查询转换为一个高维的数值向量（Language Embedding，图中所示形状为 (1, 1536)），以便网络能够理解和处理。

4. **Bilinear Resize (白色方框)**:

   **对应代码**: 在 `inference.py` 的 `predict` 方法的最后阶段，网络输出的低分辨率相似度图 `sim_np` 被调整回原始图像的大小 `(H, W)`。这通过 `torchvision.transforms.functional.resize` 实现，并明确指定了插值模式：`interpolation=T.InterpolationMode.BILINEAR` 8。

   **含义**: 由于网络核心部分处理的是降采样后的特征图，网络的输出也是低分辨率的。这一步使用双线性插值将预测的 affordance 图放大到与原始输入图像相同的分辨率，以便进行可视化和叠加。

### 图表正确性评估

**这个图片是正确的。**

它准确地概述了 `inference.py` 和 `network.py` 中定义的推理流程：

**输入**: 正确展示了图像和文本的双重输入 9。

**特征提取**: 准确反映了图像经过预处理后由冻结的 DINO 模型提取特征，文本经过嵌入函数处理的过程 10。

**核心网络**: 中心的 `Conv2DFiLMNet` 块及其 `FiLM Conditioning` 机制正确地表示了图像特征和语言嵌入在 `network.py` 中定义的融合方式 11。

**后处理**: Sigmoid 激活、可选的阈值处理以及最后的双线性调整大小都与 `predict` 方法中的代码逻辑一一对应 12。

**数据形状**: 图中标记的中间数据形状（如 Image Features 的 (1, 1024, h, w) 和 Language Embedding 的 (1, 1536)）也与代码中的配置和操作一致 13。



