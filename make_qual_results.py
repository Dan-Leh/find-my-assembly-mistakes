# %% Testing model on real data - v2 -> more samples, and references corresponding to correct states
import torch
from torchvision.transforms import v2
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import types
import os
import numpy as np

from datasets.synthtestset import EvalDataset
from models.build_functions import build_model

# MODEL_NAMES = ["gca_from_VISION", "gca", "gca_fda50%_mildaug", "gca_fda75%_mildaug", "gca_fda75%_moreaug", "gca_fda75%_noaug", "gca_fda50%_moreaug"]
# MODEL_NAMES = ["gca_from_VISION", "noam_final", "gca_final", "msa_final", "lca_final"]
# MODEL_NAMES = ["noam_final", "gca_final", "msa_final", "lca_final"]
# MODEL_NAMES = ["gca_like_VISION_ft", "gca_fda50%_moreaug", "gca_rand_bg_50%", "gca_final"]  # no shear, no shear, shear, shear

MODEL_NAMES = ["aligned_noam30%randbg", "aligned_gca30%randbg", "noam_final", "gca_final"]

NUM_SAMPLES = 20
TEST_TYPE = 'roi'

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229,0.224,0.225]
detransform = v2.Compose([
    v2.Normalize(mean = [ 0., 0., 0. ],
                std = [1/val for val in norm_std]),
    v2.Normalize(mean = [-val for val in norm_mean],
                std = [ 1., 1., 1. ]),
    v2.ToPILImage()
])

############# get data

dataroot = "/shared/nl011006/res_ds_ml_restricted/dlehman/SyntheticData/Val_states_new_parts"
data_list_filepath = os.path.join(dataroot, 'eval_rand_30k_w_crops.json')

dataset = EvalDataset(data_list_filepath = data_list_filepath, 
            test_type = TEST_TYPE) # make roi aligned for aligned images

anchors = []
samples = []
labels = []
i=0
while(True):
    j = np.random.randint(0, dataset.__len__())
    image1, image2, changemask, nqd, maxparts = dataset.__getitem__(j)
    
    if nqd > 0:
        anchors.append(image1)
        samples.append(image2)
        labels.append(changemask)
        i+=1
        
        if i == NUM_SAMPLES:
            break
    
    
fig, axs = plt.subplots(NUM_SAMPLES, 3+len(MODEL_NAMES), figsize=(5*len(MODEL_NAMES), 3*1.2*NUM_SAMPLES))

for model_idx, model_name in enumerate(MODEL_NAMES):

    config_path = f"/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/{model_name}/config.yaml"
    with open(config_path, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        configfile.close()

    config_ns = types.SimpleNamespace(**config)
    model, device = build_model(args = config_ns)
    checkpoint_dir = config_ns.checkpoint_dir
    if model_name.startswith('noam'):
        checkpoint = torch.load(os.path.join(checkpoint_dir, "last_ckpt.pt"), map_location='cpu')
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "best_ckpt.pt"), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    save_root = os.path.join("/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results", 
                             f"{model_name}", "unaligned_novel_parts")

    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            
            image1 = anchors[i]
            image2 = samples[i]
            gt = labels[i]
            
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
                gt = v2.ToPILImage()(gt)
                gt_blend = Image.blend(image1.convert("RGBA"), gt.convert("RGBA"), 0.8)
                axs[i,2].imshow(gt_blend);  axs[i,model_idx+3].axis('off')
                

            prediction = v2.ToPILImage()(torch.squeeze(prediction,0))
            blend = Image.blend(image1.convert("RGBA"), prediction.convert("RGBA"), 0.8)
            axs[i,model_idx+3].imshow(blend);  axs[i,model_idx+3].axis('off')
            
            # save images
            if save_root!="":
                if not os.path.exists(save_root): os.mkdir(save_root)
                image1.save(os.path.join(save_root, f"{i}_ref.png"))
                image2.save(os.path.join(save_root, f"{i}_sample.png"))
                blend.save(os.path.join(save_root, f"{i}_prediction.png"))
                gt_blend.save(os.path.join(save_root, f"{i}_gt.png"))
    
    axs[0,model_idx+3].set_title(model_name)
axs[0,0].set_title('Reference')
axs[0,1].set_title('Sample')
axs[0,2].set_title('Ground Truth')

plt.tight_layout()
# plt.savefig("/shared/nl011006/res_ds_ml_restricted/dlehman/state-diff-net/results/comparison_fda.png")

# %% Plot histogram of normalized weights

# fig, axs = plt.subplots(3, 1, figsize=(5, 8))

# for i in range(3):
#     axs[i].hist(rgb_synth[i], color='blue', label='synth', alpha=0.5, bins=50)
#     axs[i].hist(rgb_real[i], color='orange', label='real', alpha=0.5, bins=50)
#     axs[i].legend()
#     axs[i].set_title(f'Distribution for channel {i}')
