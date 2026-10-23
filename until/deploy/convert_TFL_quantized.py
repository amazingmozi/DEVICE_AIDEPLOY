# convert_model_quantized.py
import tensorflow as tf
import numpy as np

# 加载模型
model = tf.keras.models.load_model('checkpoints/sine_model.h5')

# 创建转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 设置优化选项
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 提供代表性数据集用于量化
def representative_dataset():
    """生成代表性数据集"""
    for _ in range(100):
        data = np.random.uniform(0, 2*np.pi, (1, 1)).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# 设置为全整数量化
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 转换模型
tflite_quant_model = converter.convert()

# 保存量化模型
with open('sine_model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)


print(f"Quantized model size: {len(tflite_quant_model)} bytes")
