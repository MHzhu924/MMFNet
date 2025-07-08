# MMFNet
This project provides the code and results for 'Mamba and Multi-domain Fusion Network for 360° Salient Object Detection'

# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/ACCoNet/blob/main/image/ACCoNet.png">
   </div>
   
   
# Requirements
   python 2.7 + pytorch 0.4.0 or
   
   python 3.7 + pytorch 1.9.0


# Saliency maps
   We provide saliency maps of our ACCoNet ([VGG_backbone](https://pan.baidu.com/s/11KzUltnKIwbYFbEXtud2gQ) (code: gr06) and [ResNet_backbone](https://pan.baidu.com/s/1_ksAXbRrMWupToCxcSDa8g) (code: 1hpn)) on ORSSD, EORSSD, and additional [ORSI-4199](https://github.com/wchao1213/ORSI-SOD) datasets.
      
   ![Image](https://github.com/MathLee/ACCoNet/blob/main/image/table.png)
   
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
                
                
