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

def load_anchor_img(anchor_path):
    ''' load anchor and apply transforms '''
    
    # dataset = SyntheticChangeDataset(data_path="/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData/IndustReal_states_v2",
    #                                  img_transforms = { "ROI_crops": False,
    #                                                     "brightness": 0,
    #                                                     "center_roi": False,
    #                                                     "contrast": 0,
    #                                                     "frac_random_background": 0,
    #                                                     "g_kernel_size": 1,
    #                                                     "g_sigma_h": 1,
    #                                                     "g_sigma_l": 1,
    #                                                     "gradually_augment": False,
    #                                                     "hflip_probability": 0,
    #                                                     "hue": 0,
    #                                                     "img_size": [256,256],
    #                                                     "max_translation": 0,
    #                                                     "normalization": 'imagenet',
    #                                                     "random_crop": False,
    #                                                     "rescale": 1.3,
    #                                                     "rotation": False,
    #                                                     "saturation": 0,
    #                                                     "shear": 0,
    #                                                     "vflip_probability": 0},
    #                                  fda_config = {"frac_imgs_w_fda": 0},
    #                                  orientation_thresholds = (0,0),
    #                                  parts_diff_thresholds = (0,0)                          
    #                                  )
    
    # idx = sequence * dataset.n_frames + frame
    # data = dataset.__getitem__(idx)
    # anchor = data[0]
    
    img = Image.open(anchor_path).convert("RGB")
    
    anchor_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(256),
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
    

def imgs_to_video(img_list, save_path, video_filename = 'created_video.mp4', fps=30):
    video_path = os.path.join(save_path, video_filename)
    
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(video_path, codec, fps, (512,256))
    
    for img in img_list:
        vid_writer.write(img)

    vid_writer.release()
    
    
#%%

MODEL_NAMES = ["lca_final"]
NUM_SAMPLES = 'all'
ANCHOR_PATH = "/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net-extra/videos/anchor.png"
VIDEO_FRAMES_DIR = "/shared/nl011006/res_ds_ml_restricted/GouthamBalachandran/error_demo/rgb"
SAVE_PATH = "/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net-extra/videos"
VIDEO_NAME = "lca_final.mp4"
FPS = 15

anchor = load_anchor_img(ANCHOR_PATH)
anchor_detrans = detransform(anchor)

all_frames = os.listdir(VIDEO_FRAMES_DIR)
all_frames = sorted([frame for frame in all_frames if frame.endswith('.jpg')])
if NUM_SAMPLES == 'all':
    frame_names = all_frames
else:
    frame_names = all_frames[:NUM_SAMPLES]

# fig, axs = plt.subplots(NUM_SAMPLES, 2+len(MODEL_NAMES), figsize=(5*len(MODEL_NAMES), 2*NUM_SAMPLES))

predictions = []

for model_idx, model_name in enumerate(MODEL_NAMES):

    model, device = load_model(model_name)
    
    with torch.no_grad():
        for i, frame in enumerate(frame_names):
            
            frame_path = os.path.join(VIDEO_FRAMES_DIR, frame)

            # load and crop image
            sample = Image.open(frame_path)
            sample = sample_transforms(sample)
                    
            prediction = model(torch.unsqueeze(anchor,0).to(device), torch.unsqueeze(sample,0).to(device))
            prediction = torch.argmax(prediction, dim=1, keepdim=True).to(torch.uint8) * 255

            print(f"Model {model_name}, image {i}")

            # Plot the images
            sample = detransform(sample)
            # axs[i,0].imshow(anchor_detrans); axs[i,0].axis("off") 
            # axs[i,1].imshow(sample); axs[i,1].axis("off")

            prediction = v2.ToPILImage()(torch.squeeze(prediction,0))
            blend = Image.blend(anchor_detrans.convert("RGBA"), prediction.convert("RGBA"), 0.8)
            # axs[i,model_idx+2].imshow(blend);  axs[i,model_idx+2].axis('off')
            
            # for making video
            npimg1 = np.array(blend.convert('RGB'))
            npimg2 = np.array(sample.convert('RGB'))
            npimg = np.hstack((npimg1,npimg2))
            npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
            predictions.append(npimg)

imgs_to_video(predictions, SAVE_PATH, VIDEO_NAME, FPS)

# axs[0,model_idx+2].set_title(model_name)
# axs[0,0].set_title('Reference')
# axs[0,1].set_title('Sample')

# plt.tight_layout()
# plt.savefig("/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/video.png")


# def create_result_video(rec_dir: Path, config: dict, pred: list, vid_load_path: Path, title="result",save_path  = None):
#     print("-"*69)
#     print(f"Creating video for {rec_dir.name}")
#     print("-" * 69)
#     name = rec_dir.name
#     # We read the fraems from rec_dir
#     frames = list((rec_dir / 'rgb').glob("*.jpg"))
#     frames.sort()
#     n_frames = len(frames)

#     # load ASD predictions
#     asd_predictions = load_asd_predictions(
#         config["ads_dir"], rec_dir, n_frames)

#     res_path = Path(save_path)
#     vid_path = res_path / f"{config['implementation']}" / f"{name}_{title}.mp4"
#     vid_path.parent.mkdir(parents=True, exist_ok=True)
#     save_video = cv2.VideoWriter(str(vid_path), fourcc, FPS, (width, height))
#     load_video = cv2.VideoCapture(str(vid_load_path))
# %%
