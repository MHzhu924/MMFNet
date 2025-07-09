# MMFNet
This project provides the code and results for 'Mamba and Multi-domain Fusion Network for 360° Salient Object Detection'

# Network Architecture
   <div align=center>
   <img src="https://github.com/MHzhu924/MMFNet/blob/main/MFFNet-main/images/MFFNet.png">
   </div>
   
# Requirements
```
Python 3.10 + PyTorch 2.3.1 (CUDA 11.8)
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install matplotlib timm  opencv-python imageio scikit-image
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
pip install causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


# Saliency maps
We provide saliency maps of our MFFNet on [360-SOD](https://pan.baidu.com/s/1sRNpJuJh1prR3D_lzf6kRA) (code: cir4) and [360-SSOD](https://pan.baidu.com/s/1iUJuq3IT44gBlPI_86t1qg) (code: 86t5) datasets .  

   
# Training
   Download [pvt_v2_b2.pth](https://pan.baidu.com/s/16I0ESyEKRDvzD828uOxyAg) (code: kfb9), and put it in './model/'. 
   Modify paths of datasets, then run train_MFFNet.py.

Note: Our main model is under './model/MFFNet_models.py' (PVT-v2-b2 backbone)


# testing
   Run test_MFFNet.py.

# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
                
                
