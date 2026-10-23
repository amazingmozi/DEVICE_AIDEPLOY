# model_to_header.py
def convert_to_c_array(tflite_model_path, output_path):
    """将TFLite模型转换为C数组"""
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()

    # 生成C头文件
    with open(output_path, 'w') as f:
        f.write('// Auto-generated file\n')
        f.write('#ifndef SINE_MODEL_DATA_H\n')
        f.write('#define SINE_MODEL_DATA_H\n\n')
        f.write('const unsigned char sine_model_data[] = {\n')

        # 每行16个字节
        for i in range(0, len(model_data), 16):
            chunk = model_data[i:i+16]
            hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
            f.write(f'  {hex_str},\n')

        f.write('};\n')
        f.write(f'const unsigned int sine_model_data_len = {len(model_data)};\n\n')
        f.write('#endif  // SINE_MODEL_DATA_H\n')

    print(f"Model converted to C array: {output_path}")
    print(f"Model size: {len(model_data)} bytes")

# 转换模型
convert_to_c_array('sine_model_quantized.tflite', 'sine_model_data.h')