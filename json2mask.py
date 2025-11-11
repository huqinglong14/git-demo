import json
import os
import cv2
import numpy as np
from pathlib import Path

def scale_points(input_path, output_path, scale_factor=10):
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历所有形状
    for shape in data['shapes']:
        # 对每个坐标点进行缩放
        scaled_points = [[x * scale_factor, y * scale_factor] for x, y in shape['points']]
        shape['points'] = scaled_points  # 更新坐标

    # 写入新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"坐标已成功放大{scale_factor}倍并保存至 {output_path}")

def json_to_mask(json_path, output_path, img_width, img_height):
    """
    将JSON标注转换为彩色Mask图像（红色填充）
    参数：
    json_path: JSON标注文件路径
    output_path: Mask图像保存路径
    img_width: 原始图像宽度
    img_height: 原始图像高度
    """
    # 创建三通道BGR画布（初始为黑色）
    mask = np.zeros((int(img_height), int(img_width), 3), dtype=np.uint8)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历所有标注形状
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon' and shape['label'] in ['mito', 'ap', 'cc']:
            # 转换坐标为NumPy数组
            pts = np.array(shape['points'], dtype=np.int32)
            # 在mask上绘制红色填充多边形
            cv2.fillPoly(mask, [pts], color=(0, 0, 255))  # 红色 (B, G, R)

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 保存mask图像
    cv2.imwrite(output_path, mask)
    print(f"Mask已保存至：{output_path}")

if __name__ == "__main__":
    # 定义路径
    images_dir = r"E:\studydie\data\all\animal\images"
    labels_dir = r"E:\studydie\data\all\animal\labels"
    scaled_labels_dir = r"E:\studydie\data\all\animal\scaled_labels"    # 正常图片只需要使用标签的的原比例即可，特殊标注（比如目标特别小，标注点需要精确到小数位）则可以放大十倍去掉小数点
    masks_dir = r"E:\studydie\data\all\animal\masks"

    # 获取图像文件列表并过滤
    images_names = [f for f in os.listdir(images_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    print(f"Images found ({len(images_names)}): {images_names}")


    # 获取JSON文件列表
    json_names = [f for f in os.listdir(labels_dir) if f.lower().endswith('.json')]
    print(f"JSONs found ({len(json_names)}): {json_names}")

    # 放大坐标
    scale_factor = 1
    os.makedirs(scaled_labels_dir, exist_ok=True)
    for input_json_name in json_names:
        input_json = os.path.join(labels_dir, input_json_name)
        output_json_name = "scaled_" + input_json_name
        output_json = os.path.join(scaled_labels_dir, output_json_name)
        scale_points(input_json, output_json, scale_factor=scale_factor)

    # 将缩放后的JSON文件可视化为掩码
    scaled_json_names = [f for f in os.listdir(scaled_labels_dir) if f.lower().endswith('.json')]
    print(f"Scaled JSONs found ({len(scaled_json_names)}): {scaled_json_names}")

    # 检查文件配对
    if len(scaled_json_names) != len(images_names):
        print(f"Warning: JSONs ({len(scaled_json_names)}) and images ({len(images_names)}) count mismatch.")
        # 仅处理配对的文件
        paired_files = min(len(scaled_json_names), len(images_names))
        scaled_json_names = scaled_json_names[:paired_files]
        images_names = images_names[:paired_files]

    for idx, (input_json_name, image_name) in enumerate(zip(scaled_json_names, images_names)):
        json_file = os.path.join(scaled_labels_dir, input_json_name)
        image_path = os.path.join(images_dir, image_name)
        mask_name = "mask_" + input_json_name.split(".")[0] + ".png"
        mask_output = os.path.join(masks_dir, mask_name)

        # 读取图像并获取尺寸
        print(f"Processing [{idx}]: {image_name} with {input_json_name}")
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            original_height, original_width = img.shape[:2]
            json_to_mask(json_file, mask_output, scale_factor * original_width, scale_factor * original_height)
        else:
            print(f"Failed to read image: {image_path}. Check file format, corruption, or permissions.")