# Low-dose 3D cone-beam CT (CBCT) reconstruction challenge

## About
  - Team member names:
    - Sasidhar Alavala (MS(R), IIT Tirupati, India)
    - Dr. Subrahmanyam Gorthi (Asst. Professor, IIT Tirupati, India)
  - Abstract:
    - Our approach integrates SwinIR-based sinogram enhancement module, coupled with Nesterov Accelerated Gradient Descent (NAG) for solving the least squares problem in CT image reconstruction. To address the challenge of excessive blurring during reconstruction, a second phase of image processing is done. This involves using another SwinIR-based CT enhancement module for enhancing features that may have been compromised in the reconstruction process. The combination of sinogram enhancement and CT enhancement modules aims to provide a better solution for low-dose and clinical-dose CBCT reconstruction, offering improved image clarity and fine detail preservation.

## Model Zoo

Please go to [MODEL HUB](https://drive.google.com/drive/folders/1VZyLtVW9JpTRaNqLXnzITJCXM7Qjjc1T?usp=sharing) for model weights.

## Usage

### Install

You can use either the [conda environment file](conda_environment.yml) to install dependencies or create a new environment and install the mentioned packages:

- Creating conda test enivronment with YML file:

```bash
conda env create -f conda_environment.yml
```

- Creating conda test enivronment without YML file:

```bash
conda create -n test_env python==3.10.12
conda activate test_env
```
Then install `torch==2.1.0+cu121` `torchvision==0.16.0+cu121` `cudatoolkit=11.3.1` `numpy==1.26.0` `timm==0.9.12` `astra-toolbox=2.1.2` `tomosipo==0.6.0` `ts-algorithms==0.1.0`

### Data preparation
The data and model weights folder structure is as follows:
  ```bash
  $ tree data
  data
  ├── sino_test_low
  │   ├── 0901_sino_low_dose.npy
  │   ├── 0902_sino_low_dose.npy   
  │   └── ...
  ├── sino_test_clinical
  │   ├── 0901_sino_clinical_dose.npy
  │   ├── 0902_sino_clinical_dose.npy   
  │   └── ...
  ├── ct_groundtruth
  │   ├── 0901_clean_fdk_256.npy
  │   ├── 0902_clean_fdk_256.npy   
  │   └── ...
  ├── ct_output_low
  ├── ct_output_clinical
  └── model_zoo
      ├── low_sino_231.pth
      ├── low_ct_117.pth
      ├── clinical_sino_148.pth   
      └── clinical_ct_186.pth
 
  ```
### Evaluation
To evaluate on the test dataset run:

```bash
python3 test_low.py
python3 test_clinical.py
```

## References
- Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1833-1844).
- Hendriksen, A. A., Schut, D., Palenstijn, W. J., Viganó, N., Kim, J., Pelt, D. M., ... & Batenburg, K. J. (2021). Tomosipo: fast, flexible, and convenient 3D tomography for complex scanning geometries in Python. Optics Express, 29(24), 40494-40513.
- Nesterov, Y. (2003). Introductory lectures on convex optimization: A basic course (Vol. 87). Springer Science & Business Media.

## License and Acknowledgement
This work is made public under the MIT license. The codes are based on [SwinIR](https://github.com/JingyunLiang/SwinIR) and [ts_algorithms](https://github.com/ahendriksen/ts_algorithms). Please also follow their licenses. Thanks for their awesome works.

