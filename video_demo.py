# %% Testing model on real data - v2 -> more samples, and references corresponding to correct states
import torch
from torchvision.transforms import v2
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import types
import os
import random
import cv2
import numpy as np

from datasets.synthdataset import SyntheticChangeDataset
from models.build_functions import build_model


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]
detransform = v2.Compose([
    v2.Normalize(mean = [ 0., 0., 0. ],
                std = [1/val for val in norm_std]),
    v2.Normalize(mean = [-val for val in norm_mean],
                std = [ 1., 1., 1. ]),
    v2.ToPILImage()
])

def load_model(model_name):
    # load config
    config_path = f"/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/{model_name}/config.yaml"
    if model_name == "gca_from_VISION":
        config_path = f"/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca/config.yaml"
    with open(config_path, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        configfile.close()

    config_ns = types.SimpleNamespace(**config)
    model, device = build_model(args = config_ns)
    if "checkpoint_dir" not in config.keys():
        checkpoint_dir = os.path.join(config["checkpoint_root"], config["project_name"])
    else:
        checkpoint_dir = config_ns.checkpoint_dir
    if model_name.startswith('ft_') or model_name.startswith('noam'):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "last_ckpt.pt"), map_location='cpu')
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "best_ckpt.pt"), map_location='cpu')
    if model_name == "gca_from_VISION":
        checkpoint_path = "/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net-extra/ckpts_from_vision/gca/best_ckpt.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print(device)
    
    return model, device

anchor_config = {"error_run_assembly_pulley":{
                            "anchors": [
                                "seq0113step0009.camera.png",
                                "seq0111step0012.camera.png",
                                "seq0010step0012.camera.png",
                                "seq0134step0012.camera.png" ],
                            "start_frames": np.array([0, 350, 450, 510])
                            },
                "error_run_disassembly_frontwheels":{
                            "anchors": [
                                "seq0134step0012.camera.png",
                                "seq0116step0011.camera.png" ],
                            "start_frames": np.array([0, 280])
                            },
                "error_run_assembly_frontbracket":{
                            "anchors": ["anchor.png"],
                            "start_frames": np.array([0])
                            }
                }


def load_anchor_img(anchor_root, error_run, frame, highres):
    ''' load anchor and apply transforms '''
    
    anchor_cfg = anchor_config[error_run]
    idx = np.max(np.where(anchor_cfg["start_frames"]<=frame))
    anchor_name = anchor_cfg["anchors"][idx]
    
    anchor_path = os.path.join(anchor_root, anchor_name)
    
    img = Image.open(anchor_path).convert("RGB")
    
    anchor_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(256) if not highres else v2.Resize(720),
            v2.Normalize(mean=norm_mean, std=norm_std)
            ])
    
    anchor = anchor_transforms(img)
    
    return anchor
    
    
sample_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(720),
            v2.Resize(256),
            v2.Normalize(mean=norm_mean, std=norm_std)
            ])

sample_transforms_hr = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop(720),
            v2.Normalize(mean=norm_mean, std=norm_std)
            ])
    

def imgs_to_video(img_list, save_path, video_filename, fps, video_img_size):
    video_path = os.path.join(save_path, video_filename)
    
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_path, codec, fps, video_img_size)
    
    for img in img_list:
        vid_writer.write(img)

    vid_writer.release()
    
    
def make_visualizations(filtered_preds, preds, samples, anchors, only_filtered):
    ''' blend anchor & prediction, and arrange frames side-by-side for video. '''
    
    def blend_w_anchor(pred, anchor):
        # convert from white to red
        pred = np.expand_dims(pred,-1)  # R
        zeros = np.zeros((256,256,2))   # G & B
        pred = np.concatenate((pred, zeros), axis=2).astype(np.uint8)
        pred = Image.fromarray(pred * 255) 
        if anchor.size != pred.size:
            pred = pred.resize(anchor.size, Image.NEAREST)         
        img = Image.blend(anchor, pred.convert("RGBA"), 0.5)
        img = np.array(img.convert('RGB'))
        return img
    
    output = []
    for fpred, pred, sample, anchor in zip(filtered_preds, preds, samples, anchors):
        
        anchor_detrans = anchor.convert("RGBA")
        filtered_pred = blend_w_anchor(fpred, anchor_detrans)
        pred = blend_w_anchor(pred, anchor_detrans)
        
        # stack frames horizontally
        if only_filtered:
            npimg = np.hstack((filtered_pred, sample))
        else:
            npimg = np.hstack((pred, filtered_pred, sample))
        
        npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
        output.append(npimg)

    return output

    
def majority_voting(frames: list, window_size: int) -> np.ndarray:
    ''' 
    Arguments:
        frames (list): list of predictions 
        window_size (int): window size across which majority is required
        
    Returns: 
        filtered_frames (np array): frames after filtering 
    '''
    majority = window_size//2
    frames = np.array(frames)
    filtered_frames = np.zeros_like(frames)
    for i in range(frames.shape[0]):
        if i < window_size:
            filtered_frames[i] = frames[i]
        else:
            video_slice = frames[i-window_size:i]
            majorities = (np.sum(video_slice, axis=0)>majority)
            filtered_frames[i] = (majorities.astype(np.uint8))
    
    return filtered_frames
        

def soft_voting(frames: list, window_size: int) -> np.ndarray:
    ''' 
    Arguments:
        frames (list): list of softmax activations 
        window_size (int): window size across which average activation needs
            to exceed threshold
        
    Returns: 
        filtered_frames (np array): frames after filtering
    '''
    half_window = window_size//2
    threshold = window_size*0.8
    frames = np.array(frames)
    filtered_frames = np.zeros_like(frames)
    for i in range(frames.shape[0]):
        if i < half_window:
            filtered_frames[i] = frames[i]
        else:
            video_slice = frames[i-half_window:i+half_window]
            filtered = (np.sum(video_slice, axis=0)>threshold)
            filtered_frames[i] = (filtered.astype(np.uint8))
    
    return filtered_frames

    
#%%

ERROR_RUNS = [ "error_run_disassembly_frontwheels", "error_run_assembly_pulley"] #["error_run_assembly_frontbracket", "error_run_assembly_pulley", "error_run_disassembly_frontwheels", "error_run_pin_orientation", "error_run_scattered"]
MODEL_NAMES = ["gca_final_roim50"] #["gca_final", "msa_final", "noam_final", "gca_final_roim20", "gca_final_roim30", "gca_final_roim40", "gca_final_roim50", "gca_from_VISION",  "lca_final"]
NUM_SAMPLES = 'all'
FPS = 20
WINDOW_SIZE = 16  # window across which to take majority vote
SAVE_PREDICTIONS = False  # either run inference and save predictions, or load saved predictions
SHOW_FINAL = True  # if true, save images in higher resolution, and without raw output (only filtered)
if SHOW_FINAL: 
    assert SAVE_PREDICTIONS == False
    IMG_DIMS = (2*720, 720)
else:
    IMG_DIMS = (3*256, 256)
FILTER_METHOD = 'soft'  # either 'soft' or 'hard' for soft/hard-voting
for ERROR_RUN in ERROR_RUNS:
    VIDEO_FRAMES_DIR = "/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/data/dan_test_rgb_only/"+ERROR_RUN+"/rgb"
    SAVE_PATH = "/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net-extra/videos/"+ERROR_RUN
    for MODEL_NAME in MODEL_NAMES:
        VIDEO_NAME = MODEL_NAME+"_soft16_PRESENTATION.mp4"
        
        print(f"Making video {ERROR_RUN} with model {MODEL_NAME}")

        if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)

        save_soft_pred_path = os.path.join(SAVE_PATH, 'soft_predictions', MODEL_NAME)
        if not os.path.exists(save_soft_pred_path) and SAVE_PREDICTIONS: 
            os.makedirs(save_soft_pred_path)

        all_frames = os.listdir(VIDEO_FRAMES_DIR)
        all_frames = sorted([frame for frame in all_frames if frame.endswith('.jpg')])
        if NUM_SAMPLES == 'all':
            frame_names = all_frames
        else:
            frame_names = all_frames[:NUM_SAMPLES]

        soft_predictions = []
        predictions = []
        samples = []
        anchors = []
        if SAVE_PREDICTIONS: model, device = load_model(MODEL_NAME)
        with torch.no_grad():
            for i, frame in enumerate(frame_names):
                if np.mod(i, 100) == 0:
                    print(f"Model {MODEL_NAME}, image {i}/{len(frame_names)}")
                frame_path = os.path.join(VIDEO_FRAMES_DIR, frame)
                soft_pred_path = os.path.join(save_soft_pred_path, frame.replace('.jpg','.npy'))
                                
                # load sample image
                sample = Image.open(frame_path)
                
                
                if SAVE_PREDICTIONS:  # make model prediction
                    anchor = load_anchor_img(SAVE_PATH, ERROR_RUN, i, highres=False)
                    sample = sample_transforms(sample)
                    raw_prediction = model(torch.unsqueeze(anchor,0).to(device), torch.unsqueeze(sample,0).to(device))
                    second_channel = raw_prediction[0,1]
                    soft_prediction = second_channel.cpu().numpy()
                    np.save(soft_pred_path, soft_prediction)

                else:  # load saved predictions
                    soft_prediction = np.load(soft_pred_path)
                    anchor = load_anchor_img(SAVE_PATH, ERROR_RUN, i, highres=True)
                    sample = sample_transforms_hr(sample)
                
                anchors.append(detransform(anchor))
                samples.append(np.array(detransform(sample)))
                soft_predictions.append(np.array(soft_prediction))
                prediction = ((soft_prediction)>0.5).astype(np.uint8)
                # assert np.array_equal(softpredthresh.cpu().numpy(), prediction)
                predictions.append(np.array(prediction))

        if FILTER_METHOD == 'soft':
            filtered_predictions =soft_voting(predictions, WINDOW_SIZE)
        elif FILTER_METHOD == 'hard':
            filtered_predictions = majority_voting(predictions, WINDOW_SIZE)
        else:
            raise NotImplementedError("Invalid filter method")

        visualized_frames = make_visualizations(filtered_predictions, predictions, samples, anchors, SHOW_FINAL)
        imgs_to_video(visualized_frames, SAVE_PATH, VIDEO_NAME, FPS, IMG_DIMS)

