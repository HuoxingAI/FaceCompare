import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
import torch
# from inception_resnet import InceptionResnetV1
# from mobilenet import MobileNetV1
from resnet import resnet_face18
from iresnet import iresnet100
import os




class Facenet_V2(nn.Module):
    def __init__(self,model_path = 'model/model.pth'):
        super(Facenet_V2, self).__init__()
        self.backbone = iresnet100()
        if os.path.exists(model_path):
            self.backbone.load_state_dict(torch.load(model_path))
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_path,'model/model.pth')
            self.backbone.load_state_dict(torch.load(model_path))
        


    def forward(self, x1,x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x = torch.cosine_similarity(x1, x2,dim=1)
        x = x/2 +0.5
    
        return x

    


if __name__=="__main__":
    model = Facenet_V2()
    print(model)
    model.eval()
    img = torch.rand((4,1,112,112))
    out = model(img,img)

    print(out.shape)