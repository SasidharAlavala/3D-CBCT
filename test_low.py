#OS: Ubuntu 22.04.3 LTS
#Author: Sasidhar Alavala (mail: ansr2510@gmail.com)
################################################### Imports ########################################################
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from swinir import SwinIR
import tomosipo as ts
from ts_algorithms import nag_ls

################################################### Folder path & parameters ##########################################
b_size = 1
noisy_files_test = [f"/media/ee22s501/HDD/data/sino_test_low/{i:04d}_sino_low_dose.npy".format(i) for i in range(801,901)]
output_folder = '/media/ee22s501/HDD/data/ct_output_low/'
folder_ct = '/media/ee22s501/HDD/data/ct_groundtruth/'
model_path_1 = '/media/ee22s501/HDD/data/model_zoo/low_sino_231.pth'
model_path_2 = '/media/ee22s501/HDD/data/model_zoo/low_ct_117.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = [300, 300, 300]
image_shape = [256, 256, 256]
voxel_size = [1.171875, 1.171875, 1.171875]
detector_shape = [256, 256]
detector_size = [600, 600]
pixel_size = [2.34375, 2.34375]
dso = 575
dsd = 1050

angles = np.linspace(0, 2*np.pi, 360, endpoint=False)

vg = ts.volume(shape=image_shape, size=image_size)
pg = ts.cone(angles=angles, shape=detector_shape, size=detector_size, src_orig_dist=dso, src_det_dist=dsd)
A = ts.operator(vg, pg)

################################################### Utils #######################################################

class DenoisingDataset(Dataset):
    def __init__(self, noisy_files, transform=None):
        self.noisy_files = noisy_files
        self.transform = transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = np.load(self.noisy_files[idx]).transpose(0,2,1)

        if self.transform:
            noisy = self.transform(noisy)

        return noisy
    
def calculate_mse(a, b):
    mse = np.mean((a - b) ** 2)
    return mse

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

################################################### Load model & dataset ########################################################
best_model_1 = SwinIR(img_size=(256, 256), in_chans=360, embed_dim=90,
               depths=[6, 6, 6, 6, 6 , 6], num_heads=[6, 6, 6, 6, 6 , 6], window_size=8,
               upscale=1, img_range=1., resi_connection='3conv', mlp_ratio=2)
best_model_1 = torch.nn.DataParallel(best_model_1, device_ids=[0])
best_model_1.load_state_dict(torch.load(model_path_1)['model_state_dict'])
best_model_1.to(device)

best_model_2 = SwinIR(img_size=(256, 256), in_chans=256, embed_dim=90,
               depths=[6, 6, 6, 6, 6 , 6], num_heads=[6, 6, 6, 6, 6 , 6], window_size=8,
               upscale=1, img_range=1., resi_connection='3conv', mlp_ratio=2)
best_model_2 = torch.nn.DataParallel(best_model_2, device_ids=[0])
best_model_2.load_state_dict(torch.load(model_path_2)['model_state_dict'])
best_model_2.to(device)

test_dataset = DenoisingDataset(noisy_files_test, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False)

################################################### Test loop ########################################################
total_mse1 = 0.0
with torch.no_grad():
    for i, (noisy) in enumerate(test_loader):
        noisy = noisy.to(device)/700
        denoised = best_model_1(noisy)
        denoised = denoised.cpu().numpy().squeeze()

        filename = noisy_files_test[i].split('/')[-1]
        denoised_sino = denoised.transpose(1,0,2)*700

        sino = torch.from_numpy(denoised_sino).cuda()
        recon_n = nag_ls(A, sino, num_iterations=18, max_eigen=106742.3828125)

        recon_n = recon_n.to(device)
        recon_n = recon_n.permute(2,0,1)
        recon_n = recon_n.unsqueeze(0)

        recon = best_model_2(recon_n)
        recon = recon.cpu().numpy().squeeze()
        recon = recon.transpose(1,2,0)
 
        filename_ct = filename.replace('sino_low_dose', 'clean_fdk_256')       
        clean = np.load(folder_ct + filename_ct, allow_pickle=True)
        mse1 = calculate_mse(recon, clean)
        total_mse1 += mse1.item()

        filename_save = filename.replace('sino', 'ct')
        np.save(output_folder + filename_save, recon)

average_mse1 = total_mse1 / len(test_loader)
print('Number of test samples: {}'.format(len(test_loader)))
print('Average MSE on test dataset: {:.15f}'.format(average_mse1))
