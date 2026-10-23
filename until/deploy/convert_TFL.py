# 实现对.h5模型的转换为.tflite模型的功能
from pathlib import Path

import tensorflow as tf


# 1. 设置输入的 Keras .h5 模型路径和输出的 .tflite 模型路径
h5_path = Path("sine_model.h5")
tflite_path = Path("sine_model.tflite")

# 2. 加载已经训练好的 .h5 模型
model = tf.keras.models.load_model(h5_path)

# 3. 创建 TensorFlow Lite 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 4. 执行模型转换，得到 .tflite 二进制内容
tflite_model = converter.convert()

# 5. 确保输出目录存在，并保存 .tflite 模型文件
tflite_path.parent.mkdir(parents=True, exist_ok=True)
tflite_path.write_bytes(tflite_model)

# 6. 打印转换完成后的模型保存路径
print(f"TFLite model saved to: {tflite_path}")
print(f"Original model size: {len(tflite_model)} bytes")

