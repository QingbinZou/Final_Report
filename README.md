![image](https://github.com/user-attachments/assets/5f339b0e-8625-4f1a-85be-6dfff41417d9)
U-Net 的特征提取进行一系列卷积 + 池化操作：
前面几层提取局部细节（如边缘、角点）
中间几层提取中层结构（如线条、物体轮廓）
越往后越关注全局和语义信息（比如识别是一根柱子）
-------
ResBlock增强特征学习能力，捕捉更复杂的 RS 伪影模式
-------
增强了特征学习能力，能捕捉更复杂的 RS 伪影模式。
-------
L1 Loss	tf.reduce_mean(tf.abs(y_true - y_pred))	精确还原图像像素值，减少整体差异；  
SSIM Loss	1 - tf.reduce_mean(tf.image.ssim(...))	强调图像的结构相似性，提高感知质量；  
Gradient Loss	使用 sobel_edges 获取梯度，约束图像边缘清晰度	防止模糊，保持清晰轮廓和结构；  
-------
![image](https://github.com/user-attachments/assets/8561c238-1357-4e2b-be9a-8061c418d488)

https://github.com/user-attachments/assets/ce63fa56-7649-4a3a-aa60-5d2653e3ac4e

https://github.com/user-attachments/assets/ff7c8798-acdd-4925-b12d-446ac3ebb765

忽略的一个错误（训练和测试使用同一组数据）
-------

