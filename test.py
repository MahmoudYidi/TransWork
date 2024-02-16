# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py (Potential Change)
"""
import math
import os
import sys

sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
from bbox3d import BBox3D
import torch
from bounding import *
from depth_net.model import *
import util.misc as utils
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from torchvision import transforms
from models import build_model
from datasets.mydata import make_mydata_transforms
from dbbox import *
import matplotlib.pyplot as plt
import time
import math


def euler_from_quaternion(rot):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = rot[0]
    y = rot[1]
    z = rot[2]
    w = rot[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return np.array(
        [math.degrees(roll_x), math.degrees(pitch_y), math.degrees(yaw_z)]
    )  # in radians


def convert_2d_point(vec):
    K = np.array([[322.28, 0, 320.8], [0, 322.28, 178.779], [0, 0, 1]])
    # xyz is a 3d point in camera coordinates
    xyz = vec
    # project into homogeneous image coordinates
    uvw = K.dot(xyz)
    # pixel coordinates
    uv = uvw[:2] / uvw[2]
    return uv


def loading_depth_img(filename):
    img = Image.open(filename)
    image = np.array(img)
    transform = Compose([ToPILImage(), Resize((480, 640)), ToTensor()])
    image = transform(image)
    image = image.to(device)
    return image


def rescale_center(a, size_old, size_new):
    w_old, h_old = size_old
    x = round((a[0] / w_old) * size_new[0])
    y = round((a[1] / h_old) * size_new[1])

    return [x, y]


def recover_rot(rot):
    rot = np.array(rot)
    rx = rot[0][0][0]
    ry = rot[0][0][1]
    rz = rot[0][0][2]
    rw = rot[0][0][3]

    return [rx, ry, rz, rw]


def recover_trans(trans):
    trans = np.array(trans)
    tx = trans[0][0][0]
    ty = trans[0][0][1]
    tz = trans[0][0][2]

    return [tx, ty, tz]


def compute_new_poses(tz, fx, fy, px, py, cx, cy):
    tz = tz / 10000
    tx = ((cx - px) * tz) / fx
    ty = ((cy - py) * tz) / fy

    return [tx, ty, tz]


def recover_centre_w_h(x, size):
    img_w, img_h = size
    final = x * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return final


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    # print(x_c,y_c,w,h)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_images(in_path):
    img_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
            ):
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=1, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="mydata")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir",
        default="",
        help="path where to save the results, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--thresh", default=0.5, type=float)

    return parser


@torch.no_grad()
def infer(images_path, model, i2d, postprocessors, device, output_path):
    model.eval()
    i2d.eval()
    print(images_path)
    duration = 0
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
        print(img_sample)
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        transform = make_mydata_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        start_t = time.perf_counter()
        outputs = model(image)
        pred_depth = loading_depth_img(image_paths[0])
        pred_depth = i2d(pred_depth[None, ...])

        # print(pred_depth)
        end_t = time.perf_counter()
        # print(outputs.keys())
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        outputs["translation"] = outputs["translation"].cpu()
        outputs["rotation"] = outputs["rotation"].cpu()
        # print(outputs['rotation'])
        print(outputs["translation"])

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        # print([outputs['pred_boxes']])

        coupled_bbox = recover_centre_w_h(
            outputs["pred_boxes"], orig_image.size
        )  # Gives you cx, cy, w, h

        coupled_bbox.cpu().data.numpy()
        # coupled_bbox = coupled_bbox.astype(np.float)
        coupled_bbox = np.array(coupled_bbox)
        center_x = round(coupled_bbox[0][0][0])
        center_y = round(coupled_bbox[0][0][1])
        # print(coupled_bbox[0])
        # center_x = coupled_bbox[]
        rotation = recover_rot(outputs["rotation"])
        translation = recover_trans(outputs["translation"])
        print(translation)
        # print(rotation)

        bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], orig_image.size)

        # print(scaled_rot)
        probas = probas[keep].cpu().data.numpy()

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features["0"].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            continue

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for idx, box in enumerate(bboxes_scaled):
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            print("data ", bbox[0])
            bbox = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ]
            )
            # print(bbox[0])
            # print(bbox[1])
            # print(bbox[2])
            # print(bbox[3])
            bbox = bbox.reshape((4, 2))
            bbox1 = bboxcorrect(bbox, 10, 10)

            print("bb =", bbox)

            cv2.polylines(img, [bbox1], True, (0, 255, 0), 2)
            bbox2 = bboxcorrect(bbox, 0, -10)
            cv2.polylines(img, [bbox2], True, (0, 255, 0), 2)
            print(bbox1[0][0])
            # cv2.polylines(img, [bbox2], True, (0, 255, 0), 5)
            cv2.line(img, bbox1[0], bbox2[0], (0, 255, 0), 2)
            cv2.line(img, bbox1[1], bbox2[1], (0, 255, 0), 2)
            cv2.line(img, bbox1[2], bbox2[2], (0, 255, 0), 2)
            cv2.line(img, bbox1[3], bbox2[3], (0, 255, 0), 2)
            # drawing(img,euler_from_quaternion(rotation),np.array(translation))
            # bbox_3d = BBox3D(translation[0],translation[1],translation[2],length=0.1, width = 0.1, rx=rotation[0], ry=rotation[1], rz=rotation[2], rw=rotation[3])

        # img_save_path = os.path.join(output_path, filename)
        # cv2.imwrite(img_save_path, img)
        # cv2.imshow("img", img)
        cv2.imwrite("test.png", img)
        depth_img = transforms.ToPILImage()(pred_depth.cpu().int().squeeze(0))
        depth_img.save("pred.png")
        depth_img = np.array(depth_img)
        # print(depth_img)
        print(center_x, center_y)
        scaled_centers = rescale_center(
            [center_x, center_y],
            orig_image.size,
            [depth_img.shape[1], depth_img.shape[0]],
        )

        # Rx, Ry, Rz = euler_from_quaternion(rotation)
        # Rx1, Ry1, Rz1 = euler_from_quaternion(-0.0070372949,	0.3362813064,	-0.031702534,	0.9412015252)

        # print(Rx, Ry, Rz)
        # print(Rx1, Ry1, Rz1)
        # print(math.degrees(Rx), math.degrees(Ry), math.degrees(Rz))
        # print(math.degrees(Rx1), math.degrees(Ry1), math.degrees(Rz1))

        # depth_img.save("pred.png")
        tz = depth_img[scaled_centers[1], scaled_centers[0]]
        print(tz)
        new_pose = compute_new_poses(
            tz, 607.499, 607.42, 325, 231.3, center_x, center_y
        )
        print(new_pose)

        cv2.imwrite("depth.png", depth_img)
        # depth_img = transforms.ToPILImage()(pred_depth.cpu().int().squeeze(0))
        # pred_depth = np.array(pred_depth)
        # cv2.imwrite('depth.png', pred_depth[0][0][0])
        # cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    image_paths = get_images(args.data_path)

    LOAD_DIR = "."
    i2d = I2D().to(args.device)
    i2d.load_state_dict(
        torch.load("depth_net/fyn_model.pt".format(LOAD_DIR), map_location="cpu")
    )
    print("model loaded")

    infer(image_paths, model, i2d, postprocessors, device, args.output_dir)
