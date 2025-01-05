import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from WindowsMultiheadAttention import WindowsMultiheadAttention
from ShiftWindowsMultiheadAttention import ShiftWindowsMultiheadAttention
import patch_merging

class SwinTransformerBlock(nn.Module):
    """
    4个MLP,4个LayerNorm
    WindowAttention+ShiftWindowAttention
    """
    def __init__(self,model_dim,window_size=7,num_heads=8,flatten_output=False):
        super(SwinTransformerBlock, self).__init__()

        self.flatten_output=flatten_output

        self.layer_norm1=nn.LayerNorm(model_dim)
        self.layer_norm2=nn.LayerNorm(model_dim)
        self.layer_norm3=nn.LayerNorm(model_dim)
        self.layer_norm4=nn.LayerNorm(model_dim)

        self.wmsa_mlp1=nn.Linear(model_dim,4*model_dim)
        self.wmsa_mlp2=nn.Linear(4*model_dim,model_dim)

        self.swmsa_mlp1=nn.Linear(model_dim,4*model_dim)
        self.swmsa_mlp2=nn.Linear(4*model_dim,model_dim)

        self.wmsa=WindowsMultiheadAttention(model_dim=model_dim,window_size=window_size,num_heads=num_heads)
        self.swmsa=ShiftWindowsMultiheadAttention(model_dim=model_dim,window_size=window_size,num_heads=num_heads)

    def forward(self,x):
        """
        input:x[bs,nums_patch,patch_depth] 类似序列
        return:sequence [bs,nums_patch,patch_depth] or window [bs,nums_window,nums_patch_in_window,patch_depth]
        """
        bs,nums_patch,patch_depth=x.shape
        # 层归一化+WMSA
        wmsa_input1=self.layer_norm1(x)
        wmsa_input1,_=self.wmsa(wmsa_input1)# [bs,nums_window, nums_patch_in_window,patch_depth]

        # 残差
        bs,nums_window,nums_patch_in_window,patch_depth=wmsa_input1.shape # 获取形状
        wmsa_input1=wmsa_input1.reshape(bs,nums_patch,patch_depth)
        wmsa_output1=wmsa_input1+x# [bs,nums_patch,patch_depth]
        
        # MLPx2＋残差
        wmsa_input2=self.wmsa_mlp2(self.wmsa_mlp1(self.layer_norm2(wmsa_output1)))
        wmsa_output2=wmsa_input2+wmsa_output1# [bs,nums_patch,patch_depth]
        
        # 层归一化+SWMSA
        swmsa_input1=self.layer_norm3(wmsa_output2)
        swmsa_input1=swmsa_input1.reshape(bs,nums_window,nums_patch_in_window,patch_depth)# [bs,nums_window, nums_patch_in_window,patch_depth]
        swmsa_input1,_=self.swmsa(swmsa_input1)
        swmsa_input1=swmsa_input1.reshape(bs,nums_patch,patch_depth)
        # 残差
        swmsa_output1=swmsa_input1+wmsa_output2

        # MLPx2＋残差
        swmsa_input2=self.swmsa_mlp2(self.swmsa_mlp1(self.layer_norm4(swmsa_output1)))
        swmsa_output2=swmsa_input2+swmsa_output1

        # 如果不展开，返回窗口类型
        if not self.flatten_output:
            swmsa_output2=swmsa_output2.reshape(bs,nums_window,nums_patch_in_window,patch_depth)

        return swmsa_output2 
    
def test():
    bs,nums_patch,patch_depth=32,3136,96
    input=torch.randn(32,3136,96)
    flatten_output=True
    model=SwinTransformerBlock(model_dim=patch_depth,window_size=7,num_heads=8,flatten_output=flatten_output)
    output=model(input)
    print(output.shape)
if __name__=='__main__':
    test()




        
