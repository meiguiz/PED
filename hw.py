import cv2
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import numpy as np
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return yL,y_HL,y_LH,y_HH,x
input=cv2.imread('/media/meiguiz/HIKVISION/PED/TransCG-TONG/hwd/rgb2.png')
input=np.transpose(input,(2,0,1))
input=np.expand_dims(input,axis=0)
input=torch.FloatTensor(input).to('cpu')
model=Down_wt(in_ch=3,out_ch=3)
a,b,c,d,e=model(input)

a=a.squeeze(0).cpu().detach().numpy()
b=b.squeeze(0).cpu().detach().numpy()
c=c.squeeze(0).cpu().detach().numpy()
d=d.squeeze(0).cpu().detach().numpy()
e=e.squeeze(0).cpu().detach().numpy()

a=np.transpose(a,(1,2,0))
b=np.transpose(b,(1,2,0))
c=np.transpose(c,(1,2,0))
d=np.transpose(d,(1,2,0))
e=np.transpose(e,(1,2,0))

# a=np.clip(a,0.0,1.0)
# a=(a*255).astype(np.uint8)
a=a.astype(np.uint8)
b=np.clip(b,0.0,1.0)
b=(b*255).astype(np.uint8)
c=np.clip(c,0.0,1.0)
c=(c*255).astype(np.uint8)
d=np.clip(d,0.0,1.0)
d=(d*255).astype(np.uint8)
e=np.clip(e,0.0,1.0)
e=(e*255).astype(np.uint8)
cv2.imwrite('a.png',a)
cv2.imwrite('b.png',b)
cv2.imwrite('c.png',c)
cv2.imwrite('d.png',d)
cv2.imwrite('e.png',e)

