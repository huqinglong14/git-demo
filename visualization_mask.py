from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def visualizetion(source_path, target_path):
    image_names = os.listdir(source_path)
    for image_name in image_names:
        # 打开.tif图片
        image_path = os.path.join(source_path, image_name)
        image = Image.open(image_path)



        # 将图片转换为 NumPy 数组
        image_array = np.array(image)
        image_array*= 255

        unique,count = np.unique(image_array,return_counts=True)

        unique_values, counts = np.unique(image_array, return_counts=True)
        # # 显示图片（可选）
        # img = Image.fromarray(image_array)
        # img.show()

        # 将 numpy 数组转换为 PIL 图像
        image = Image.fromarray(image_array)

        # 保存图像为 PNG 文件
        image.save(os.path.join(target_path, image_name.split('.')[0]+'.png'))
        print(f"mask{image_name.split('.')[0]+'.png'} is done")


if __name__ == "__main__":
    source_path = r"D:\studydie\data\xianlitidata\images"
    target_path = r"D:\studydie\dataset\GDCLD\train_data\demo"
    visualizetion(source_path, target_path)
