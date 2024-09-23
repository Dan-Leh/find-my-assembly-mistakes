# %% Testing model on real data - v2 -> more samples, and references corresponding to correct states
import torch
from torchvision.transforms import v2
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import types
import os

from datasets.real_img_dataset import RealChangeDataset
from models.build_functions import build_model

# MODEL_NAMES = ["gca_from_VISION", "gca", "gca_fda50%_mildaug", "gca_fda75%_mildaug", "gca_fda75%_moreaug", "gca_fda75%_noaug", "gca_fda50%_moreaug"]
# MODEL_NAMES = ["gca_from_VISION", "noam_final", "gca_final", "msa_final", "lca_final"]
# MODEL_NAMES = ["noam_final", "gca_final", "msa_final", "lca_final"]
# MODEL_NAMES = ["gca_like_VISION_ft", "gca_fda50%_moreaug", "gca_rand_bg_50%", "gca_final"]  # no shear, no shear, shear, shear

MODEL_NAMES = ["noam_final", "lca_final"]

NUM_SAMPLES = 47



norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]
detransform = v2.Compose([
    v2.Normalize(mean = [ 0., 0., 0. ],
                std = [1/val for val in norm_std]),
    v2.Normalize(mean = [-val for val in norm_mean],
                std = [ 1., 1., 1. ]),
    v2.ToPILImage()
])

fig, axs = plt.subplots(NUM_SAMPLES, 2+len(MODEL_NAMES), figsize=(5*len(MODEL_NAMES), 3*1.2*NUM_SAMPLES))

for model_idx, model_name in enumerate(MODEL_NAMES):
    # print(f"Model name: {model_name}")

    config_path = f"/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/{model_name}/config.yaml"
    if model_name == "gca_from_VISION":
        config_path = f"/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/gca/config.yaml"
    with open(config_path, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        configfile.close()

    dataset = RealChangeDataset(data_path = "/shared/nl011006/res_ds_ml_restricted/dlehman/IndustReal_test_imgs/v3_assy",
                                ROI = True, img_tf = config["img_transforms"])

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

    save_root = os.path.join("/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results", 
                             f"{model_name}", "real_img_results")


    # rgb_synth = [None]*3
    # rgb_real = [None]*3
    with torch.no_grad():
        for i in range(dataset.__len__()):

            image1, image2, category = dataset.__getitem__(i)
            axs[i,model_idx+2].set_title(category)
            axs[i,0].set_title(f"Ref_{i}")
            
            # if i==0:
            #     for j in range(3):
            #         rgb_synth[j] = image1[j].view(-1)
            #         rgb_real[j] = image2[j].view(-1)
            # else:
            #     for j in range(3):
            #         rgb_synth[j] = torch.cat([rgb_synth[j], image1[j].view(-1)])
            #         rgb_real[j] = torch.cat([rgb_real[j], image2[j].view(-1)])
                    
            
            prediction = model(torch.unsqueeze(image1,0).to(device), torch.unsqueeze(image2,0).to(device))
            prediction = torch.argmax(prediction, dim=1, keepdim=True).to(torch.uint8) * 255
                    
            print(f"Model {model_name}, image {i}")

            # Plot the images
            image1 = detransform(image1)
            if model_idx == 0 or save_root!="":
                image2 = detransform(image2)
                axs[i,0].imshow(image1); axs[i,0].axis("off") 
                axs[i,1].imshow(image2); axs[i,1].axis("off")
                

            prediction = v2.ToPILImage()(torch.squeeze(prediction,0))
            blend = Image.blend(image1.convert("RGBA"), prediction.convert("RGBA"), 0.8)
            axs[i,model_idx+2].imshow(blend);  axs[i,model_idx+2].axis('off')
            
            # save images
            if save_root!="" and model_name!="gca_from_VISION":
                if not os.path.exists(save_root): os.mkdir(save_root)
                image1.save(os.path.join(save_root, f"{i}_ref.png"))
                image2.save(os.path.join(save_root, f"{i}_sample.png"))
                blend.save(os.path.join(save_root, f"{i}_Prediction.png"))
    
    axs[0,model_idx+2].set_title(model_name)
axs[0,0].set_title('Reference')
axs[0,1].set_title('Sample')

plt.tight_layout()
# plt.savefig("/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/comparison_fda.png")

# %% Plot histogram of normalized weights

# fig, axs = plt.subplots(3, 1, figsize=(5, 8))

# for i in range(3):
#     axs[i].hist(rgb_synth[i], color='blue', label='synth', alpha=0.5, bins=50)
#     axs[i].hist(rgb_real[i], color='orange', label='real', alpha=0.5, bins=50)
#     axs[i].legend()
#     axs[i].set_title(f'Distribution for channel {i}')
