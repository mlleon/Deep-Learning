import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


def getClsTxt(xmlDir, cls_txt, allowed_labels):
    """
    xmlDir        : XML directory path
    cls_txt       : Output cls file path
    allowed_labels: List of allowed labels
    """

    invalid_label_paths = []  # List to store paths of XML files with invalid labels

    for name in tqdm(os.listdir(xmlDir)):
        xmlFile = os.path.join(xmlDir, name)
        with open(xmlFile, "r+", encoding='utf-8') as fp:
            tree = ET.parse(fp)
            root = tree.getroot()

            invalid_labels = set()
            for obj in root.iter('object'):
                cls_element = obj.find('name')
                if cls_element is not None:
                    cls = cls_element.text
                    invalid_labels.add(cls)
                    if cls not in allowed_labels:
                        invalid_label_paths.append((xmlFile, cls))  # Store both XML path and invalid label

            set_cls.update(invalid_labels)

    if invalid_label_paths:
        print("Invalid labels found in the following XML files:")
        for path, invalid_label in invalid_label_paths:
            print(f"{path}, Error category is: {invalid_label}")
    else:
        print("No invalid labels found.")

    with open(cls_txt, "w+") as ft:
        for i in set_cls:
            ft.write(i + "\n")


if __name__ == '__main__':
    set_cls = set()
    xmlDir = "F:\gitlocal\dl_code\large_files\dataset\VOCdevkit\VOC2012\Annotations"
    cls_txt = "./check_labels.txt"
    allowed_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                      '12', '13', '14', '15', '16', '17', '18', '19', '20']

    getClsTxt(xmlDir, cls_txt, allowed_labels)



