import cv2
import numpy as np
from PIL import Image
import os

class MaskConverter:
    def __init__(self, source_path, target_path):
        """
        初始化转换器
        :param source_path: 输入mask文件夹路径
        :param target_path: 输出标签文件夹路径
        """
        self.source_path = source_path
        self.target_path = target_path
        self._prepare_output_dir()

    def _prepare_output_dir(self):
        """创建输出目录"""
        os.makedirs(self.target_path, exist_ok=True)

    def convert_all(self):
        """
        转换目录下所有图像文件
        """
        valid_ext = ('.tiff', '.png', '.jpg')
        for filename in os.listdir(self.source_path):
            if filename.lower().endswith(valid_ext):
                self._process_single_image(filename)

    def _process_single_image(self, filename):
        """
        处理单个图像文件
        :param filename: 图像文件名
        """
        # 读取图像
        image_path = os.path.join(self.source_path, filename)
        image = Image.open(image_path)
        image_array = np.array(image)

        # 准备输出文件
        output_file = self._get_output_path(filename)
        open(output_file, 'w').close()  # 清空已有内容

        # 处理不同图像类型
        if image_array.ndim == 3:
            self._process_color_mask(image_array, output_file)
        else:
            self._process_grayscale_mask(image_array, output_file)

        print(f"处理完成：{filename}")

    def _get_output_path(self, filename):
        """
        生成输出文件路径
        :param filename: 输入文件名
        :return: 完整输出路径
        """
        basename = os.path.splitext(filename)[0]
        return os.path.join(self.target_path, f"{basename}.txt")

    def _process_color_mask(self, image_array, output_file):
        """
        处理彩色mask图像
        :param image_array: 三维numpy数组
        :param output_file: 输出文件路径
        """
        # 获取所有唯一颜色（排除背景黑色）
        unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
        filtered_colors = [c for c in unique_colors if not np.all(c == 0)]

        for color in filtered_colors:
            binary = self._create_binary_mask(image_array, color=color)
            label = self.color_to_label(color)
            # 跳过未定义颜色
            if label is None:
                continue

            self._process_contours(binary, output_file, self.color_to_label(color))

    def _process_grayscale_mask(self, image_array, output_file):
        """
        处理灰度mask图像
        :param image_array: 二维numpy数组
        :param output_file: 输出文件路径
        """
        unique_labels = np.unique(image_array)
        filtered_labels = [l for l in unique_labels if l != 0]

        for label in filtered_labels:
            binary = self._create_binary_mask(image_array, label=label)
            self._process_contours(binary, output_file, label)

    def _create_binary_mask(self, image_array, color=None, label=None):
        """
        创建二值掩码
        :param image_array: 输入图像数组
        :param color: 目标颜色（RGB三元组）
        :param label: 目标标签（灰度值）
        :return: 二值化后的numpy数组
        """
        if color is not None:
            return np.all(image_array == color, axis=-1).astype(np.uint8) * 255
        elif label is not None:
            return (image_array == label).astype(np.uint8) * 255    # 括号内的意思是生成一个布尔矩阵，astype(np.uint8)表示将布尔值转化成0和1
        raise ValueError("必须指定color或label参数")

    def _process_contours(self, binary, output_file, label):
        """
        处理轮廓并写入文件
        :param binary: 二值图像
        :param output_file: 输出文件路径
        :param label: 类别标签
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = binary.shape[:2]

        for contour in contours:
            approx = self._approximate_contour(contour)
            if approx is not None:
                self._write_contour(approx, output_file, label, width, height)

    def _approximate_contour(self, contour):
        """
        多边形近似处理
        :param contour: 原始轮廓
        :return: 近似后的多边形坐标
        """
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx if len(approx) >= 3 else None

    def _write_contour(self, polygon, output_file, label, width, height):
        """
        将多边形坐标写入文件
        :param polygon: 多边形坐标
        :param output_file: 输出文件路径
        :param label: 类别标签
        :param width: 图像宽度
        :param height: 图像高度
        """
        with open(output_file, 'a') as f:
            f.write(f"{label} ")
            for point in polygon.squeeze():
                x = point[0] / width
                y = point[1] / height
                f.write(f"{x:.6f} {y:.6f} ")
            f.write("\n")

    def color_to_label(self, color):
        """
        颜色到标签的映射（需根据实际情况重写）
        :param color: RGB颜色元组
        :return: 整型类别标签
        """
        # 示例映射关系
        color_mapping = {
            #tuple([255, 0, 8]): 0,    # 红色 -> 1
            tuple([255, 0, 0]):  1,    # 绿色 -> 2
           # tuple([0, 255, 247]): 0,    # 蓝色 -> 3
        }
        label = color_mapping.get(tuple(color))
        if label is None:
            print(f"[警告] 检测到未定义颜色: {color}，已跳过处理")
        return label
        #return color_mapping.get(tuple(color), 0)   # # 未定义的颜色返回0（背景


if __name__ == '__main__':
    # 使用示例
    converter = MaskConverter(
        source_path=r"D:\studydie\data\622\final_data\masks",
        target_path=r"D:\studydie\data\622\final_data\labels"
    )
    converter.convert_all()