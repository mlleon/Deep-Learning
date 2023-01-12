import os
import random


def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = "./VOCdevkit/VOC2012/Annotations"  # 存储VOC数据.xml文件根目录
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    train_txt = "./VOCdevkit/VOC2012/ImageSets/Main/train.txt"  # 存储分割VOC数据集后train.txt文件路径
    val_txt = "./VOCdevkit/VOC2012/ImageSets/Main/val.txt"  # 存储分割VOC数据集后val.txt文件路径

    val_rate = 0.3

    # files_name=['1', '2', '3', '4', 'a', 'd', 'f', 'g', 'h']
    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open(train_txt, "x")
        eval_f = open(val_txt, "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
