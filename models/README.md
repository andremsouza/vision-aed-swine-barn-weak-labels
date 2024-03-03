# Models

We provide models trained on the aSwine dataset as per our publication:

1. [AST (0.850 AUC, 0.752 mAP)](https://drive.google.com/file/d/1pr6-q536F9E6l86LwYLRdVoF_RDNOa0S/view?usp=drive_link)
2. [Inception V3 (0.848 AUC, 0.755 mAP)](https://drive.google.com/file/d/1pxwbrcg7KQHu_0E_finwb__mroK6q4mZ/view?usp=drive_link)
3. [AlexNet (0.841 AUC, 0.733 mAP)](https://drive.google.com/file/d/1q3eAhEToTmmHljR1u29iYF-ibbE-K8gs/view?usp=drive_link)
4. [VGG (0.840 AUC, 0.609 mAP)](https://drive.google.com/file/d/1ptXqxhHM8gLaWR5fghD5gSXG9Dk3dQI-/view?usp=drive_link)
5. [ResNet-50 (0.822 AUC, 0.687 mAP)](https://drive.google.com/file/d/1pnXs23BohebVaRYsvDVLRG1mCfCZ23M1/view?usp=drive_link)
6. [Fully Connected, 6 layers, 8192 hidden units per layer (0.817 AUC, 0.679 mAP)](https://drive.google.com/file/d/1q7jag0Hh-FgZnafeftw2GAlI9j_Xoox1/view?usp=drive_link)
7. [Fully Connected, 5 layers, 8192 hidden units per layer (0.811 AUC, 0.735 mAP)](https://drive.google.com/file/d/1qBMDyXMBsb-5wV9v2n1mSEfScEFi_YUa/view?usp=drive_link)

These models are provided in the form of PyTorch state dictionaries. To use them, you will need to load the model architecture and then load the state dictionary. For example, to load the Inception V3 model:

```python  
from inception_v3 import InceptionV3
import torch

# ...
model = InceptionV3(num_classes=7, dropout=0.5)
model.load_state_dict(torch.load('inception_v3.pth'))
# ...  
```
