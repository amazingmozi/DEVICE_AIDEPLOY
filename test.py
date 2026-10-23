# 验证量化后的 sine_model_quantized.tflite 模型效果
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset_generate import create_train_test_data


# 1. 生成和训练时一致的测试数据
_, _, x_test, y_test = create_train_test_data(
    train_samples=1000,
    test_samples=100,
    noise_std=0.1,
    seed=0,
)
# x_test = np.linspace(0, 2 * np.pi, 100).astype(np.float32)
x_test = x_test.reshape(-1)
y_test = np.sin(x_test).astype(np.float32)

# 2. 加载量化后的 TFLite 模型，并分配张量内存
interpreter = tf.lite.Interpreter(model_path="sine_model_quantized.tflite")
interpreter.allocate_tensors()

# 3. 获取模型输入和输出张量信息，包括 int8 量化参数
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_detail["quantization"]
output_scale, output_zero_point = output_detail["quantization"]

# 4. 逐条输入测试数据，完成输入量化、模型推理、输出反量化
predictions = []
for x in x_test:
    quantized_x = x / input_scale + input_zero_point
    quantized_x = np.array([[quantized_x]], dtype=np.float32)
    quantized_x = np.clip(quantized_x, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_detail["index"], quantized_x)
    interpreter.invoke()

    quantized_y = interpreter.get_tensor(output_detail["index"])
    y = (quantized_y.astype(np.float32) - output_zero_point) * output_scale
    predictions.append(y[0][0])

predictions = np.array(predictions, dtype=np.float32)

# 5. 计算模型误差指标
errors = y_test - predictions
mse = np.mean(errors ** 2)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(mse)

# 6. 计算容差准确率：预测误差小于等于 0.15 的样本比例
tolerance = 0.15
accuracy = np.mean(np.abs(errors) <= tolerance)

# 7. 打印量化模型的输入输出信息和评估结果
print(f"Input dtype: {input_detail['dtype']}")
print(f"Input scale: {input_scale}, zero_point: {input_zero_point}")
print(f"Output dtype: {output_detail['dtype']}")
print(f"Output scale: {output_scale}, zero_point: {output_zero_point}")
print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Tolerance accuracy (abs error <= {tolerance}): {accuracy:.4f}")

# 可视化结果：先按 x 从小到大排序，避免随机测试点被折线乱连
sort_index = np.argsort(x_test)
x_plot = x_test[sort_index]
y_plot = y_test[sort_index]
pred_plot = predictions[sort_index]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='True sin(x)', linewidth=2)
plt.plot(x_plot, pred_plot, 'r--', label='TFLite prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('TFLite Model Verification')
plt.legend()
plt.grid(True)
plt.savefig('tflite_verification.png')
plt.close()

# 8. 打印前 10 个样本的真实值和预测值，直观看量化模型效果
print("\nSample predictions:")
for i in range(10):
    print(
        f"x={x_test[i]:.4f}, "
        f"true={y_test[i]:.4f}, "
        f"pred={predictions[i]:.4f}, "
        f"error={abs(y_test[i] - predictions[i]):.4f}"
    )
