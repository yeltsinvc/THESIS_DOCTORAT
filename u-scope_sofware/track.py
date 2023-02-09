import numpy as np
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import shutil
import platform
import os
import argparse
from deep_sort_pytorch.deep_sort import DeepSort

from deep_sort_pytorch.utils.parser import get_config
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.datasets import LoadImages, LoadStreams
import sys
from collections import defaultdict

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import exists
#from sqlalchemy.orm import sessionmaker
from model import Position, Velocity, Object,Base
import shapely.speedups
if shapely.speedups.available:
    shapely.speedups.enable()

import shapely
from shapely.geometry import Point, LineString
import fiona
from Scripts import utilscv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, './yolov5')

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None,classVeh = None,  offset=(0, 0)):
    for i, box in enumerate(bbox):
        class_names = [c.strip() for c in open('coco.names').readlines()]
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id_track = int(identities[i]) if identities is not None else 0
        id_class = int(classVeh[i]) if classVeh is not None else 0
        class_name = class_names[id_class]
        color = compute_color_for_labels(id_track)
        #label = '{}{:d}'.format(id, int(id_class))
        label = class_name+"-"+str(id_track)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4),
                      color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, session, save_img=False, polylines=None):
    from _collections import deque
    pts = [deque(maxlen=30) for _ in range(1000)]
    identities = []
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith(
        'http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights,
                       map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Load mask
    mask_yolo = cv2.imread('mask_yolo.png', 0)
    mask_visualization = cv2.imread('mask_visualization.png', 0)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = False
        dataset = LoadImages(source, img_size=imgsz, mask=mask_yolo)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    
    ###Get Lines Yeltsin
    count_lines,polygon_mask=utilscv.getCounterLines('videoframe.jpg')
    lines= [LineString(l.line) for l in count_lines]
    
    """###Get Lines Pablito
    polygon_shape_dict = {}
    polygon_geometry_dict = {}
    with fiona.open('polygons.json') as collection:
        for i, rec in enumerate(collection, 1):
            polygon_shape_dict[i] = shapely.geometry.shape(rec['geometry'])
            polygon_geometry_dict[i] = rec['geometry']

    polygon_mask = np.zeros((720, 1024, 3), dtype=np.uint8)
    # Draw polygons
    polygon_pts_list = []
    for polygon in polygon_geometry_dict.values():
        polygon_pts = np.array(polygon['coordinates'][0], dtype=int)
        polygon_pts.shape = (-1, 1, 2)
        polygon_pts_list.append(polygon_pts)
    cv2.polylines(polygon_mask, polygon_pts_list, True, (255, 255, 255), 10)"""

    y1 = 240
    y2 = 480
    region_dict = defaultdict(list)
    # region 0: not exist
    # region 1: y < y1
    # region 2: y1 < y < y2
    # region 3: y2 < y

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if frame_idx %2 ==0:
            continue
        """if frame_idx>500:
            break"""

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        delete=[]
        for bb in pred:
            pedestrian_bbox=bb[bb[:,5]==0]
            velo_bbox=bb[bb[:,5]==1]
            moto_bbox=bb[bb[:,5]==3]
            if len(pedestrian_bbox) and (len(velo_bbox) or len(moto_bbox)):
                for ped in pedestrian_bbox:
                    for velo in velo_bbox:                        
                        iou_velo=utilscv.bb_intersection_over_union(ped[:4],velo[:4])
                        if iou_velo>0.2:
                            delete.append(ped[1])
                    for k,moto in enumerate(moto_bbox):
                        iou_velo=utilscv.bb_intersection_over_union(ped[:4],moto[:4])
                        if iou_velo>0.2:
                            delete.append(ped[1])
                for delet in delete:
                    bb=bb[bb[:,1]!=delet]
            if len(bb[bb[:,5]==1])>0:
                bb[bb[:,5]==1,1] =  bb[bb[:,5]==1,1] -0.5*(bb[bb[:,5]==1,3] -bb[bb[:,5]==1,1] )
            if len(bb[bb[:,5]==3])>0:
                bb[bb[:,5]==3,1] =  bb[bb[:,5]==3,1] -0.5*(bb[bb[:,5]==3,3] -bb[bb[:,5]==3,1] )
        pred=[bb]
        
        #iou_velo=[for i,j zip(pred[:,-2],pred[:,1:])]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                cls_list = []

                # Adapt detections to deep sort input format

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    cls_list.append(cls)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs= deepsort.update(xywhs, confss, im0, cls_list)

                t3 = time_synchronized()
                #print("Deepsort {}".format(t3 - t2))

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    draw_boxes(im0, bbox_xyxy,identities,outputs[:, 5])
                    for ind, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        indx = output[4]

                        center = (int((bbox_left + bbox_w) / 2),
                                  int((bbox_top + bbox_h) / 2))

                        base = (int((bbox_left + bbox_w) / 2), int(bbox_h))
                        lines_bbox=[]
                        lines_bbox.append(LineString([(bbox_left,bbox_top),(bbox_left,bbox_h)]))
                        lines_bbox.append(LineString([(bbox_left,bbox_top),(bbox_w,bbox_top)]))
                        lines_bbox.append(LineString([(bbox_w,bbox_h),(bbox_w,bbox_top)]))
                        lines_bbox.append(LineString([(bbox_w,bbox_h),(bbox_left,bbox_h)]))
                        
                        col_count_line=99
                        for n,count_line in enumerate(lines):
                            for line_bbox in lines_bbox:
                                if not indx in count_lines[n].id_vehs:
                                    if line_bbox.intersection(count_line):
                                        count_lines[n].id_vehs.append(indx)
                                        count_lines[n].count+=1
                                        col_count_line=n
                                        #print("Line "+str(n)+" "+str(count_lines[n].count))
                                        #input('Presiona enter: ')
                                        if not session.query(exists().where(Object.object_id == int(indx))).scalar():
                                            obj = Object(
                                                object_id = int(indx),
                                                road_user_type = int(output[5]),
                                                
                                            )
                                            session.add(obj)
                                        break
                        """"
                        new_shape_id = 0
                        for id_shape, shape in polygon_shape_dict.items():
                            if shape.contains(Point(base)):
                                new_shape_id = id_shape
                                break

                        print(new_shape_id)
                        if region_dict[indx]:
                            prev_shape = region_dict[indx][-1]
                            if prev_shape != new_shape_id and prev_shape != 0 and new_shape_id != 0:
                                print(f'{indx=} {prev_shape=} {new_shape_id=}')
                                input('Presiona enter: ')

                        region_dict[indx].append(new_shape_id)"""
                        

                        position = Position(
                            trajectory_id=int(indx),
                            frame_number=int(frame_idx),
                            x_coordinate=base[0],
                            y_coordinate=base[1],
                            line_n=col_count_line
                            
                        )
                        session.add(position)
                        
                        

                        pts[indx].append(center)
                        for j in range(1, len(pts[indx])):
                            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                            color = compute_color_for_labels(indx)
                            cv2.line(im0, (pts[indx][j - 1]), (pts[indx][j]),
                                     color, thickness)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') %
                                    (frame_idx, identity, bbox_left, bbox_top,
                                     bbox_w, bbox_h, -1, -1, -1,
                                     -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            #print('Yolo {}'.format(t2 - t1))

            # Stream results
            if view_img:
                # Apply mask
                im1 = cv2.add(im0, polygon_mask)
                #im1 = cv2.bitwise_and(im1, im1, mask=mask_visualization)

                cv2.imshow(p, im1)
                print(f"{im1.shape=}")
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release(
                            )  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc),
                            fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='yolov5/weights/yolov5l.pt',
                        help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source',
                        type=str,
                        default='inference/images',
                        help='source')
    parser.add_argument('--output',
                        type=str,
                        default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.4,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.5,
                        help='IOU threshold for NMS')
    parser.add_argument('--fourcc',
                        type=str,
                        default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='filter by class')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort",
                        type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args(["--source", "video", "--save-txt"])
    args.img_size = check_img_size(args.img_size)
    #print(args)

    engine = create_engine('sqlite:///data.db')
    #Session = sessionmaker()
    #Session.configure(bind=engine)
    
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with torch.no_grad():
        #session=Session()
        with Session(engine) as session:
            detect(args, session)
            session.commit()
