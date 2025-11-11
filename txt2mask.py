import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_polygons(file_path):
    polygons = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # 忽略无效行
            category_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append((category_id, polygon))
    return polygons

def create_mask(image_shape, polygons, category_to_color=None):
    if category_to_color is None:
        category_to_color = {i: (0, 0, 255) for i in range(10)}  # 默认颜色映射

    mask = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    for category_id, polygon in polygons:

        scaled_polygon = np.array([(int(x * image_shape[1]), int(y * image_shape[0])) for x, y in polygon], dtype=np.int32).reshape((-1, 1, 2))
        color = category_to_color.get(category_id, (0, 0, 255))  # 默认红色
        cv2.fillPoly(mask, [scaled_polygon], color)

    return mask

def creat_bbox(bboxes, image_shape):
    image = np.zeros((image_shape[0],image_shape[1], image_shape[2]),dtype=np.uint8)
    for bbox in bboxes:
        id = bbox[0]
        top_left = (bbox[1], bbox[2])
        bottom_right = (bbox[3], bbox[4])
        top_left = tuple(map(int, top_left))  # 尝试将它们转换为整数元组
        bottom_right = tuple(map(int, bottom_right))
        color = (0, 255, 0)  # 绿色
        thickness = 2  # 线条粗细
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    cv2.imwrite('output_with_rectangle.jpg', image)
    return image


def parse_bbox(file_path,width,height):
    list = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            category_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            W, H = width,height
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H
            list.append([category_id, x1, y1, x2, y2])
    return list

def overlay_mask(image, mask, alpha=0.5):
    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape")

    overlaid_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    return overlaid_image

if __name__ == '__main__':

    # 读取原始图像
    original_image = cv2.imread(r"D:\studydie\data\splitxianliti2\images\image2_h01c2_4_3.tiff")
    # 假设原始图像的大小为 (height, width, channels)
    image_shape = (original_image.shape[0],original_image.shape[1], 3)  # 你可以根据实际情况调整

    # 是否是目标检测
    is_detect = False

    # 解析矩形框
    if is_detect:
        bboxes = parse_bbox(r"D:\studydie\objectdetect\datasets\coco128\labels\000000000009.txt", image_shape[1],image_shape[0])
        mask = creat_bbox(bboxes, image_shape)
        overlaid_image = overlay_mask(original_image, mask)
    else:
        # 解析多边形
        polygons = parse_polygons(r"D:\studydie\data\splitxianliti2\labels\image2_h01c2_4_3.txt")

        # 创建掩码
        mask = create_mask(image_shape, polygons)
        # 如果原始图像是灰度图像，转换为彩色图像
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # 叠加掩码
        overlaid_image = overlay_mask(original_image, mask)

        # # 显示结果
        # plt.imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # 保存叠加了掩码的图像
    cv2.imwrite('annotated_image.jpg', overlaid_image)