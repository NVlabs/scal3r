# Scal3R Overview

## Description:  
Scal3R reformulates online 3D reconstruction as multi-reference relative pose querying. A small set of learnable pose query tokens (~1% of parameters) is injected into a completely frozen backbone via asymmetric attention, and the predicted relative constraints are aggregated by online pose-graph optimization with keyframe selection and loop closure, suppressing long-range drift while fully preserving the backbone's pointmap quality.  
Scal3R was developed by NVIDIA as a part of Scal3R.  
_This model is for research and development only._  


### License/Terms of Use:
NVIDIA License
### Deployment Geography:  
Global

### Use Case:  
Developers and researchers building scalable online 3D reconstruction systems, such as SLAM (Simultaneous Localization and Mapping) or multi-view stereo pipelines, who need efficient multi-relative pose query capabilities. The model supports applications in robotics, augmented reality, and 3D mapping where real-time pose estimation from image streams is required.  
  

### Release Date:
Github 07/31/2026 via https://github.com/NVlabs/scal3r  

  

## Reference(s):
[Continuous 3D Perception Model with Persistent State](https://github.com/CUT3R/CUT3R)  
[STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer](https://github.com/NIRVANALAN/STream3R)  
[Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction](https://github.com/NVlabs/scal3r)  

## Model Architecture:   
**Architecture Type:** Transformer   
**Network Architecture:** Vision Transformer (ViT)  
**This model was developed based on CUT3R, STream3R.**  
**Number of model parameters:** Scal3R-CUT3R: 810M (8.1*10^8); Scal3R-STream3R: 1.22B (1.22*10^9); Total: 2.03B (2.03*10^9)  

## Input:  
**Input Type(s):** Video  
**Input Format(s):** RGB (Red, Green, Blue)  
**Input Parameters:** Three-Dimensional (3D)  
**Other Properties Related to Input:** The model employs sliding window attention to process long outdoor sequences, supporting up to K=12 reference frames without retraining. It uses RoPE (Rotary Position Embedding) for positional encoding, with configurable feature dimension and sequence length via the `seq_len` argument. Input tensors are expected to have even feature dimensions and are processed on CUDA devices with fallback to CPU.  
  

## Output:  
**Output Type(s):** Other: Pointcloud  
**Output Format:** Tensor  
**Output Parameters:** Three-Dimensional (3D)  
**Other Properties Related to Output:** The model returns a list of output tensors from each attention block along with an integer indicating the patch token start index. It supports streaming inference with KV cache and configurable attention modes (causal, window, full). Output tensors are normalized to the range [0,1] after patch embedding.  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** Not Applicable (N/A)    
**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Hopper
* NVIDIA Lovelace

**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
scal3r  

To integrate Scal3R, first set up a conda environment with PyTorch, torchvision, habitat-sim, and other dependencies, then compile the CUDA kernels for RoPE positional embeddings. Next, download the pretrained Scal3R checkpoint from Hugging Face (nvidia/scal3r) or via the provided download scripts, and load the model using STream3R.from_pretrained or the appropriate backbone loader. Finally, run inference or training using the supplied example notebooks and scripts, which rely on PyTorch and the Transformers library for model handling.  
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:** Image  

**Image Training Data Size:** Less than a Million Images  
**Data Collection Method by dataset:** Hybrid: Synthetic, Automatic/Sensors  
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The model is fine-tuned exclusively on TartanAir (Wang et al., 2020), a large-scale photorealistic synthetic dataset rendered in Unreal Engine/AirSim, comprising approximately 307K RGB frames from 369 camera trajectories across diverse simulated indoor and outdoor environments (urban, rural, industrial, and natural scenes) with precise ground-truth camera poses. Data consists of monocular RGB images from a simulated pinhole camera; only RGB frames and ground-truth trajectories are used for supervision (no depth or LiDAR). During training, 4-view samples (one current frame plus three reference frames) are drawn with randomly perturbed temporal intervals to cover varying motion baselines, totaling ~640K image views over 40 epochs. The dataset was publicly released in 2020; as fully synthetic data, it contains no personally identifiable information.

### Testing Dataset:

Not Applicable. No holdout split from the training dataset was used for testing; all reported results are from external zero-shot evaluation benchmarks (see Evaluation Dataset below).

### Evaluation Dataset:
| Model | KITTI (ATE) | VKITTI (ATE) | Sintel (ATE) | TUM (ATE) | ScanNet (ATE) |
| --- | --- | --- | --- | --- | --- |
| Scal3R-CUT3R | 69.7 | 5.63 | 0.168 | 0.033 | 0.092 |
| Scal3R-STream3R | 70.8 | 7.92 | 0.171 | 0.018 | 0.049 |   

**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Automatic/Sensors  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The model is evaluated on camera pose estimation benchmarks including TUM-dynamics, Sintel, ScanNet, Virtual KITTI 2, and KITTI odometry, as well as multi-view reconstruction on the 7-Scenes dataset. These datasets cover indoor and outdoor scenes with varying dynamics, lighting, and texture conditions. Evaluation follows the protocols of MonST3R and Spann3R for Sintel, TUM-dynamics, and ScanNet, and uses preprocessed versions for Virtual KITTI 2 and KITTI odometry.



## Inference:
**Acceleration Engine:** Not Applicable (N/A)
**Test Hardware:**  NVIDIA Ampere (A100 - 80GB)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.  
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

**Generated by NVIDIA Model Card Generator Toolkit.**