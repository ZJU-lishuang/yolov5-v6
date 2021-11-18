import cv2
import torch
import random
import argparse
import numpy as np
from utils.modelscfg import Darknet
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords


def copy_conv(conv_src,conv_dst):
    conv_dst[0] = conv_src.conv
    conv_dst[1] = conv_src.bn
    conv_dst[2] = conv_src.act

def copy_weight_v6(modelyolov5,model):
    focus = list(modelyolov5.model.children())[0]
    copy_conv(focus.conv, model.module_list[1])
    conv1 = list(modelyolov5.model.children())[1]
    copy_conv(conv1, model.module_list[2])
    cspnet1 = list(modelyolov5.model.children())[2]
    copy_conv(cspnet1.cv2, model.module_list[3])
    copy_conv(cspnet1.cv1, model.module_list[5])
    copy_conv(cspnet1.m[0].cv1, model.module_list[6])
    copy_conv(cspnet1.m[0].cv2, model.module_list[7])
    copy_conv(cspnet1.cv3, model.module_list[10])
    conv2 = list(modelyolov5.model.children())[3]
    copy_conv(conv2, model.module_list[11])
    cspnet2 = list(modelyolov5.model.children())[4]
    copy_conv(cspnet2.cv2, model.module_list[12])
    copy_conv(cspnet2.cv1, model.module_list[14])
    copy_conv(cspnet2.m[0].cv1, model.module_list[15])
    copy_conv(cspnet2.m[0].cv2, model.module_list[16])
    copy_conv(cspnet2.m[1].cv1, model.module_list[18])
    copy_conv(cspnet2.m[1].cv2, model.module_list[19])
    copy_conv(cspnet2.m[2].cv1, model.module_list[21])
    copy_conv(cspnet2.m[2].cv2, model.module_list[22])
    copy_conv(cspnet2.cv3, model.module_list[25])
    conv3 = list(modelyolov5.model.children())[5]
    copy_conv(conv3, model.module_list[26])
    cspnet3 = list(modelyolov5.model.children())[6]
    copy_conv(cspnet3.cv2, model.module_list[27])
    copy_conv(cspnet3.cv1, model.module_list[29])
    copy_conv(cspnet3.m[0].cv1, model.module_list[30])
    copy_conv(cspnet3.m[0].cv2, model.module_list[31])
    copy_conv(cspnet3.m[1].cv1, model.module_list[33])
    copy_conv(cspnet3.m[1].cv2, model.module_list[34])
    copy_conv(cspnet3.m[2].cv1, model.module_list[36])
    copy_conv(cspnet3.m[2].cv2, model.module_list[37])
    copy_conv(cspnet3.cv3, model.module_list[40])
    conv4 = list(modelyolov5.model.children())[7]
    copy_conv(conv4, model.module_list[41])
    spp = list(modelyolov5.model.children())[8]
    copy_conv(spp.cv1, model.module_list[42])
    model.module_list[43] = spp.m[0]
    model.module_list[45] = spp.m[1]
    model.module_list[47] = spp.m[2]
    copy_conv(spp.cv2, model.module_list[49])
    cspnet4 = list(modelyolov5.model.children())[9]
    copy_conv(cspnet4.cv2, model.module_list[50])
    copy_conv(cspnet4.cv1, model.module_list[52])
    copy_conv(cspnet4.m[0].cv1, model.module_list[53])
    copy_conv(cspnet4.m[0].cv2, model.module_list[54])
    copy_conv(cspnet4.cv3, model.module_list[56])
    conv5 = list(modelyolov5.model.children())[10]
    copy_conv(conv5, model.module_list[57])
    upsample1 = list(modelyolov5.model.children())[11]
    model.module_list[58] = upsample1
    cspnet5 = list(modelyolov5.model.children())[13]
    copy_conv(cspnet5.cv2, model.module_list[60])
    copy_conv(cspnet5.cv1, model.module_list[62])
    copy_conv(cspnet5.m[0].cv1, model.module_list[63])
    copy_conv(cspnet5.m[0].cv2, model.module_list[64])
    copy_conv(cspnet5.cv3, model.module_list[66])
    conv6 = list(modelyolov5.model.children())[14]
    copy_conv(conv6, model.module_list[67])
    upsample2 = list(modelyolov5.model.children())[15]
    model.module_list[68] = upsample2
    cspnet6 = list(modelyolov5.model.children())[17]
    copy_conv(cspnet6.cv2, model.module_list[70])
    copy_conv(cspnet6.cv1, model.module_list[72])
    copy_conv(cspnet6.m[0].cv1, model.module_list[73])
    copy_conv(cspnet6.m[0].cv2, model.module_list[74])
    copy_conv(cspnet6.cv3, model.module_list[76])
    conv7 = list(modelyolov5.model.children())[18]
    copy_conv(conv7, model.module_list[80])
    cspnet7 = list(modelyolov5.model.children())[20]
    copy_conv(cspnet7.cv2, model.module_list[82])
    copy_conv(cspnet7.cv1, model.module_list[84])
    copy_conv(cspnet7.m[0].cv1, model.module_list[85])
    copy_conv(cspnet7.m[0].cv2, model.module_list[86])
    copy_conv(cspnet7.cv3, model.module_list[88])
    conv8 = list(modelyolov5.model.children())[21]
    copy_conv(conv8, model.module_list[92])
    cspnet8 = list(modelyolov5.model.children())[23]
    copy_conv(cspnet8.cv2, model.module_list[94])
    copy_conv(cspnet8.cv1, model.module_list[96])
    copy_conv(cspnet8.m[0].cv1, model.module_list[97])
    copy_conv(cspnet8.m[0].cv2, model.module_list[98])
    copy_conv(cspnet8.cv3, model.module_list[100])
    detect = list(modelyolov5.model.children())[24]
    model.module_list[77][0] = detect.m[0]
    model.module_list[89][0] = detect.m[1]
    model.module_list[101][0] = detect.m[2]

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def img_det(model,img,img0,save_path):
    model.eval()
    pred = model(img)[0]

    pred = non_max_suppression(pred, 0.4, 0.5, classes=None,
                               agnostic=False)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (str(int(cls)), conf)
                plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)],
                             line_thickness=3)
            cv2.imwrite(save_path, img0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../cfg/yolov5s_v4.cfg', help='cfg file path')
    parser.add_argument('--path', type=str, default='../data/images/bus.jpg', help='img file path')
    parser.add_argument('--weights', type=str, default='../weights/yolov5s_v4.pt', help='sparse model weights')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading yolov5s
    modelyolov5 = torch.load(opt.weights, map_location=device)['model'].float().eval()

    from models.yolo import Detect
    inplace = True
    for m in modelyolov5.modules():
        if type(m) is Detect:
            m.inplace = inplace
            if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)

    # load yolov5s from cfg
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    copy_weight_v6(modelyolov5, model)

    path = opt.path
    img0 = cv2.imread(path)  # BGR
    # Padded resize
    img = letterbox(img0, new_shape=416)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    save_path="../v5_cfg.jpg"
    img_det(model,img, img0, save_path)
    yolov5save_path = "../v5.jpg"
    img_det(modelyolov5, img,img0, yolov5save_path)