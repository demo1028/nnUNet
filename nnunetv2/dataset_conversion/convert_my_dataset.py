import shutil
import multiprocessing
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io  # 必须引入这个库来做真正的图片格式转换

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

from PIL import Image  # 新增：引入 PIL 库专门对付 P 模式
import numpy as np  # 新增：引入 numpy 强制转换格式




def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    """
    1. 转换图像格式 (jpg -> png)
    2. 将 P 模式伪彩图 mask 转换为纯净的灰度索引图
    """
    # 1. 处理原图：读取 jpg 并保存为 png
    img = io.imread(input_image)
    io.imsave(output_image, img, check_contrast=False)

    # 2. 处理 Mask（核心去伪彩操作）：
    # 使用 PIL 读取 P 模式图像
    mask_pil = Image.open(input_seg)

    # 直接将 PIL 对象转为 numpy 数组！
    # 这一步极其神奇，它会自动丢弃伪彩调色板，只把底层的 0-6 索引提取出来
    mask_array = np.array(mask_pil, dtype=np.uint8)

    # 把剥离了伪彩外衣的纯净 0-6 数组，作为标准的 PNG 保存给 nnUNet
    io.imsave(output_seg, mask_array, check_contrast=False)


if __name__ == "__main__":
    # 1. 设置你的本地数据根目录 (绝对路径或相对路径)
    source = r'D:\DPP\graduate\Experiment\nnUNet\dataset'

    # 2. 给你的数据集按 nnUNet 规范命名 (Dataset + 3位数字 + 名字)
    dataset_name = 'Dataset501_MyToothData'

    nnUNet_raw = r"D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_raw"
    nnUNet_preprocessed = r"D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_preprocessed"
    nnUNet_results = r"D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_results"

    # 创建 nnUNet_raw 下的目标文件夹
    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # 映射你的源文件夹路径
    train_img_dir = join(source, 'train', 'img')
    train_mask_dir = join(source, 'train', 'mask')
    val_img_dir = join(source, 'test', 'img')  # 注意你上面用的是 test 文件夹
    val_mask_dir = join(source, 'test', 'mask')

    # 搜索 .jpg 格式的原图
    train_files = subfiles(train_img_dir, join=False, suffix='.jpg')
    val_files = subfiles(val_img_dir, join=False, suffix='.jpg')

    num_train = len(train_files)

    with multiprocessing.get_context("spawn").Pool(8) as p:
        r = []
        # 处理训练集
        for v in train_files:
            # v 是 "image01.jpg"
            mask_name = v.replace('.jpg', '.png')  # 寻找对应的 mask: "image01.png"
            out_img_name = v.replace('.jpg', '_0000.png')  # nnUNet 要求的原图命名
            out_mask_name = v.replace('.jpg', '.png')  # nnUNet 要求的 mask 命名

            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_img_dir, v),
                         join(train_mask_dir, mask_name),  # 使用替换后的 .png 名字去找 mask
                         join(imagestr, out_img_name),
                         join(labelstr, out_mask_name)
                     ),)
                )
            )

        # 处理验证集/测试集
        for v in val_files:
            mask_name = v.replace('.jpg', '.png')
            out_img_name = v.replace('.jpg', '_0000.png')
            out_mask_name = v.replace('.jpg', '.png')

            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(val_img_dir, v),
                         join(val_mask_dir, mask_name),
                         join(imagests, out_img_name),
                         join(labelsts, out_mask_name)
                     ),)
                )
            )
        _ = [i.get() for i in r]

    # 3. 配置数据集的 JSON 信息
    channel_dict = {0: 'R', 1: 'G', 2: 'B'}  # 假设你的牙齿 JPG 图是 RGB 的

    label_dict = {
        'background': 0,
        'class1': 1,
        'class2': 2,
        'class3': 3,
        'class4': 4,
        'class5': 5,
        'class6': 6
    }

    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        channel_dict,
        label_dict,
        num_train,
        '.png',  # 告诉 nnUNet 我们统一使用了 .png 格式
        dataset_name=dataset_name
    )
    print(f"数据集 {dataset_name} 转换并生成 dataset.json 成功！")