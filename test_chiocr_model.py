from chiocr_model import model
import glob
from PIL import Image
import numpy as np
import cv2
import re

# image_filepaths = glob.glob('/home/adam/Downloads/gsk_jpg/10/*.jpg')
image_filepaths = glob.glob('/home/adam/Pictures/201808/vat/10/*.jpg')
image_filepaths = sorted(image_filepaths)
for idx, image_filepath in enumerate(image_filepaths):
    print('Handing image {} starts: {}'.format(idx, image_filepath))
    img = Image.open(image_filepath).convert("RGB")
    img_array = np.array(img)
    img_array = img_array[:, :, ::-1]
    _, results, _ = model(img,
                          detectAngle=False,  ##是否进行文字方向检测
                          config=dict(MAX_HORIZONTAL_GAP=80,  ##字符之间的最大间隔，用于文本行的合并
                                      MIN_V_OVERLAPS=0.6,
                                      MIN_SIZE_SIM=0.6,
                                      TEXT_PROPOSALS_MIN_SCORE=0.2,
                                      TEXT_PROPOSALS_NMS_THRESH=0.3,
                                      TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                      MIN_RATIO=1.0,
                                      LINE_MIN_SCORE=0.2,
                                      TEXT_PROPOSALS_WIDTH=0,
                                      MIN_NUM_PROPOSALS=0,
                                      ),
                          leftAdjust=True,  ##对检测的文本行进行向左延伸
                          rightAdjust=True,  ##对检测的文本行进行向右延伸
                          alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                          ifadjustDegree=True
                          )
    values = []
    for result in results:
        value = result['text']
        # 去除开头的非数字字符
        value = re.sub(r'^[^\d]*', '', value)
        # 去除结尾的非数字字符
        value = re.sub(r'[^\d]*$', '', value)
        values.append(value)
    print('{}'.format(values))
    cv2.namedWindow('img_array', cv2.WINDOW_NORMAL)
    cv2.imshow('img_array', img_array)
    cv2.waitKey(0)
    print('Handing image {} ends: {}'.format(idx, image_filepath))
