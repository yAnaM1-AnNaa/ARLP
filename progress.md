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



Spatial Context: 1. The bed is not blocked by other furniture.
2. The bed is placed in the center of the room.
3. The bed is not facing a specific direction such as a TV or window.
4. Core Analysis: Considering the spatial context analyzed, the standard, primary function of this bed is to provide a flat surface for sleeping. The visible geometric structure inside the box supports this function, as the bed has a flat surface suitable for sleeping. The bed is placed in the center of the room, not blocked by other furniture, and is not facing a specific direction such as a TV or window, which further supports its primary function as a sleeping area.
