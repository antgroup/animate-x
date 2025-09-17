import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import json
import copy
import torch
import random
import argparse
import shutil
import tempfile
import subprocess
import numpy as np
import math

import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
import logging
from io import BytesIO
import oss2 as oss
import os.path as osp

import sys
import dwpose.util as util
from dwpose.wholebody import Wholebody

import pickle
from PIL import Image


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat



def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger

class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) 
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) 
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)

            # print(body.shape)
            # print(bodyfoot.shape)

            # print(body == bodyfoot[:18])

            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_body_and_foot(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas

def dw_func(_id, frame, dwpose_model, dwpose_woface_folder='tmp_dwpose_wo_face', dwpose_withface_folder='tmp_dwpose_with_face'):
    
    # frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    pose = dwpose_model(frame)

    return pose


def video2img(video_path, img_dir): 
    # pdb.set_trace()
    video_capture = cv2.VideoCapture(video_path)
    
    os.makedirs(img_dir, exist_ok=True)
    
    # Extract frames from the video
    success, image = video_capture.read()
    count = 0
    while success:
        # Save frame as JPEG file
        # cv2.imwrite(os.path.join(img_dir, f'{count:03d}.jpg'), image)
        if os.path.exists(os.path.join(img_dir, f'frame_{count:04d}.jpg')) == False:
            cv2.imwrite(os.path.join(img_dir, f'frame_{count:04d}.jpg'), image)
        
        success, image = video_capture.read()
        count += 1
        print("frame: ", count)

import cv2, os
import numpy as np
import imageio
from skimage import img_as_ubyte
def img2video(img_dir, save_path, fps = 25):
    imgs = []

    nums = sorted(os.listdir((img_dir)))
    for i in (nums):
        img_name = os.path.join(img_dir,i)

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(np.array(img))

    imageio.mimsave(save_path, [img_as_ubyte(frame) for frame in imgs], fps=fps)

def mp_main(args):
    os.makedirs(args.saved_pose_dir, exist_ok = True)
    if args.source_video_paths.endswith('mp4'):
        video_paths = [args.source_video_paths]
    else:
        # video list
        video_paths = [os.path.join(args.source_video_paths, frame_name) for frame_name in os.listdir(args.source_video_paths)]

    logger.info("There are {} videos for extracting poses".format(len(video_paths)))

    logger.info('LOAD: DW Pose Model')
    dwpose_model = DWposeDetector()  

    results_vis = []
    for i, file_path in enumerate(video_paths):
        try:



            logger.info(f"{i}/{len(video_paths)}, {file_path}")

            save_frame_dir = os.path.join(args.saved_frame_dir, os.path.basename(file_path)[:-4]) 
            os.makedirs(save_frame_dir, exist_ok = True)
            video2img(file_path, save_frame_dir)

            videoCapture = cv2.VideoCapture(file_path)
            cur_output_dir =  os.path.join(args.saved_pose, os.path.basename(file_path)[:-4]) 
            os.makedirs(cur_output_dir, exist_ok = True)
            fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
            bodies = []
            body_indices = []
            hands = []
            faces = []
            
            idx = 0
            while videoCapture.isOpened():
                # get a frame
                ret, frame = videoCapture.read()
                
                # print(frame.shape)
                # import pdb; pdb.set_trace()
                if ret:
                    size = frame.shape # (1216, 832, 3)
                    pose = dw_func(i, frame, dwpose_model)
                    bodies.append(pose['bodies']['candidate'][:18])
                    body_indices.append(pose['bodies']['subset'][0][:18])
                    faces.append(pose['faces'][0])
                    hands.append(pose['hands'])
                    # results_vis.append(pose)
                    (H,W,_) = size
                    dwpose_woface, dwpose_wface = draw_pose(
                        pose, 
                        H, 
                        W
                        # draw_face=False,
                        )
                    # output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
                    # output_transformed = cv2.resize(output_transformed, (W, H))
                        
                    # img = Image.fromarray(output_transformed)
                    cv2.imwrite(os.path.join(cur_output_dir, f"frame_{idx:04d}.jpg"), dwpose_woface)
                    # img.save(os.path.join(cur_output_dir, f"frame_{idx:04d}.jpg"))
                    idx += 1
                    # import pdb; pdb.set_trace()
                else:
                    break
            logger.info(f'all frames in {file_path} have been read.')
            videoCapture.release()

            if args.save_pose_video:
                os.makedirs(args.saved_pose+'_video', exist_ok = True)
                img2video(cur_output_dir, args.saved_pose+f'_video/'+ os.path.basename(file_path), fps)
            new_dict = {}
            new_dict['bodies'] = np.array(bodies)
            new_dict['body_indices'] = np.array(body_indices)
            new_dict['faces'] = np.array(faces)
            new_dict['hands'] = np.array(hands)
            new_dict['size'] = size
            new_dict['fps'] = fps

            
            save_pkl_path = os.path.join(args.saved_pose_dir, os.path.basename(file_path)[:-4]+'.pkl') 
            print(save_pkl_path)
            with open(save_pkl_path, 'wb') as file:
                # 使用 pickle.dump() 方法将字典写入文件
                pickle.dump(new_dict, file)

            # import pdb; pdb.set_trace()

        except:
            print(file_path," wrong")
logger = get_logger('dw pose extraction')


# python  

if __name__=='__main__':
    def parse_args(): 
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument("--source_video_paths", type=str, default="data/videos",)
        parser.add_argument("--saved_pose_dir", type=str, default="data/saved_pkl",)
        parser.add_argument("--saved_pose", type=str, default="data/saved_pose",)
        parser.add_argument("--save_pose_video", action='store_false')
        parser.add_argument("--saved_frame_dir", type=str, default="data/saved_frames",)
        args = parser.parse_args()

        return args
        
    args = parse_args()
    mp_main(args)