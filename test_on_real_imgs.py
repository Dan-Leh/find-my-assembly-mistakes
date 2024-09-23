import torch
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
import types
import os

from datasets.real_img_dataset import RealChangeDataset
from models.build_functions import build_model
from utils.manage_config import read_config

# for visualizing images:
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]
detransform = v2.Compose([
    v2.Normalize(mean = [ 0., 0., 0. ],
                std = [1/val for val in norm_std]),
    v2.Normalize(mean = [-val for val in norm_mean],
                std = [ 1., 1., 1. ]),
    v2.ToPILImage()
])

# open config
cfg = read_config(train=False)

# instantiate dataset
dataset = RealChangeDataset(data_path = os.path.join(cfg.data_root, cfg.test_set_name),
                            ROI = True, img_tf = cfg.img_transforms)

# load model
model, device = build_model(args = cfg, train=False)
print(f"Using device: {device}")

# load checkpoint
if "checkpoint_dir" not in vars(cfg).keys():
    checkpoint_dir = os.path.join(cfg.checkpoint_root, cfg.experiment_name)
else:
    checkpoint_dir = cfg.checkpoint_dir
checkpoint = torch.load(os.path.join(checkpoint_dir, "best_ckpt.pt"), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

# where to save results
save_root = os.path.join(cfg.output_dir, 'individual_results')
os.mkdir(save_root)

fig, axs = plt.subplots(dataset.__len__(), 3, figsize = (15, 150))
with torch.no_grad():
    for i in range(dataset.__len__()):
        # load image pair
        image1, image2, category = dataset.__getitem__(i)
        axs[i,2].set_title(category)
        axs[i,0].set_title(f"Anchor_{i}")
        
        # get model prediction
        prediction = model(torch.unsqueeze(image1,0).to(device), torch.unsqueeze(image2,0).to(device))
        prediction = torch.argmax(prediction, dim=1, keepdim=True).to(torch.uint8) * 255
                
        # post-process images
        image1 = detransform(image1)
        image2 = detransform(image2)
        prediction = v2.ToPILImage()(torch.squeeze(prediction,0))
        blend = Image.blend(image1.convert("RGBA"), prediction.convert("RGBA"), 0.8)
        
        # plot images
        axs[i,0].imshow(image1); axs[i,0].axis("off") 
        axs[i,1].imshow(image2); axs[i,1].axis("off")
        axs[i,2].imshow(blend);  axs[i,2].axis('off')
        
        # save images
        image1.save(os.path.join(save_root, f"{i}_ref.png"))
        image2.save(os.path.join(save_root, f"{i}_sample.png"))
        blend.save(os.path.join(save_root, f"{i}_prediction.png"))

axs[0,0].set_title('Reference')
axs[0,1].set_title('Sample')
axs[0,2].set_title('Prediction')

plt.tight_layout()
plt.savefig(os.path.join(cfg.output_dir, 'all_results.png'))

