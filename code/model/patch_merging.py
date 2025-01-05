import torch
import torch.nn as nn
import torch.nn.functional as F
from ShiftWindowsMultiheadAttention import window_to_image
import math
class patch_merging(nn.Module):
    def __init__(self,model_dim,merge_size=2):
        super(patch_merging,self).__init__()
        self.model_dim=model_dim
        self.merge_size=merge_size
        self.linear=nn.Linear(4*model_dim,2*model_dim)
    def forward(self,x):
        """
        x:window[bs,nums_patch=nums_windows*h*w,patch_depth]
        return:[bs,nums_patch//4,2*patch_depth]
        """
        
        bs,nums_window,nums_patch_in_window,patch_depth=x.shape        
        merge_size=self.merge_size

        x=window_to_image(x)# [bs,h,w,patch_depth]
        x=x.permute(0,3,1,2)# [bs,patch_depth,h,w]

        x=F.unfold(x,kernel_size=merge_size,stride=merge_size)# [bs,4*model_dim,nums_patch//4]
        x=x.transpose(1,2)
        x=self.linear(x)

        return x # [bs,nums_patch//4,model_dim*2]
    
# def test_patch_merging():
#     x=torch.randn(32,64,16,96)
#     patch_merging_layer=patch_merging(96)
#     x=patch_merging_layer(x)
#     print(x.shape)

# if __name__=='__main__':
#     test_patch_merging()
