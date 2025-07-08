# MMFNet
This project provides the code and results for 'Mamba and Multi-domain Fusion Network for 360° Salient Object Detection'

# Network Architecture
   <div align=center>
   <img src="https://github.com/MHzhu924/MMFNet/blob/main/MFFNet-main/images/MFFNet.png">
   </div>
   
# Requirements
   python 3.10.1 + pytorch 0.4.0 


# Saliency maps
   
      
   
# Training
   Download [pvt_v2_b2.pth]() (code: ), and put it in './model/'. 
   
   Modify paths of datasets, then run train_MFFNet.py.

Note: Our main model is under './model/MFFNet_models.py' (PVT-v2-b2 backbone)


# Pre-trained model and testing
1. Download the following pre-trained models and put them in /models.

2. Modify paths of pre-trained models and datasets.

3. Run test_MFFNet.py.

# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
                
                
