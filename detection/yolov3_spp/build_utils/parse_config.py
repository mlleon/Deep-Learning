import os
import numpy as np



"""
# 解析模块配置文件
1、读取配置文件每行信息 -> 去除空行和注释行 -> 去除每行开头和结尾的空格符
2、遍历lines获取block名称和block变量 -> 创建一个嵌套字典的列表[{}]用于记录所有block信息
3、先判断line是否是block名称 -> 如果是block名称，在列表mdefs最后添加一个字典记录block名称，如mdefs[-1]["type"] = line[1:-1].strip()
                         -> 如果是block变量 -> 获取block变量的键和值 -> 判断key == "anchors" -> 去除val空格 -> 逗号分割val并生成二维数组
                                                                -> 判断key in ["from", "layers", "mask"] -> 逗号分割val并生成列表
                                                                -> 其它key对应的val，若val是数值进行int处理，若val是字符串不执行处理
"""

def parse_model_cfg(path: str):     # path为yolov3-spp.cfg的路径
    # 检查文件是否存在，是否以.cfg结尾
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 读取文件信息
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # 去除空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]   # if x：表示当前行不为空
    # 去除每行开头和结尾的空格符
    lines = [x.strip() for x in lines]

    mdefs = []  # module definitions
    for line in lines:
        """遍历读取所有层结构"""
        if line.startswith("["):  # 如果line是以"["开头 -> 说明是某一个block的名称
            mdefs.append({})    # 如果是block则在mdefs列表中添加一个字典，键是type，值为block结构名称
            mdefs[-1]["type"] = line[1:-1].strip()  # 记录block类型 -> {type: block结构的名称}（line[1:-1]获取block名称）
            # 如果block是conv，设置默认不使用BN(普通conv后面会重写成1，最后的预测层conv保持为0)
            if mdefs[-1]["type"] == "convolutional":    # 由于每次都是在mdefs列表最后添加一个字典，所以mdefs[-1]取得最新添加的层结构名称
                mdefs[-1]["batch_normalize"] = 0    # 设置默认不使用BN
        else:   # 如果line不是以"["开头 -> 说明是某一个block的变量
            key, val = line.split("=")  # 使用"="对模块block变量进行分割，获取变量名(key)和变量值(val)
            # 读取进来的变量值都会自动转换为str类型，所以如果变量是数值需要再转换为对应的数据类型（int / float）
            key = key.strip()   # 变量名
            val = val.strip()   # 变量值(变量赋值内容)

            if key == "anchors":    # yolo层中的anchors
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")  # 将空格去除
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                """from是shortcut层，layers是route层"""
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                # TODO: .isnumeric() actually fails to get the float case
                # val.isnumeric()判断是否是数值的情况
                if val.isnumeric():  # return int or float 如果是数值的情况
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string  是字符的情况

    # check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    # 遍历检查每个模型的配置
    for x in mdefs[1:]:  # 0对应net配置
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))

    return mdefs


# 解析数据配置文件
def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
