import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from patch_embedding import patch_embedding
from patch_merging import patch_merging
from WindowsMultiheadAttention import WindowsMultiheadAttention
from ShiftWindowsMultiheadAttention import ShiftWindowsMultiheadAttention
from SwinTransformerblock import SwinTransformerBlock
"""
以下是模块的定义：
class patch_embedding(nn.Module):
    def __init__(self,patch_size=4,embed_size=96,nums_channel=3):
class patch_merging(nn.Module):
    def __init__(self,model_dim,merge_size=2):
class SwinTransformerBlock(nn.Module):
    def __init__(self,model_dim,window_size=7,num_heads=8):    
"""

class SwinTransformer(nn.Module):
    """
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    input:tensor of picture:[bs,channel,image_h,image_w]
    output:[bs,n_classes]
    """
    def __init__(self,patch_size=4,embed_size=96,nums_channel=3,block_depths=[2,2,6,2],merge_size=2,window_size=7,num_heads=8,n_classes=10):
        super(SwinTransformer,self).__init__()
        self.patch_embedding=patch_embedding(patch_size,embed_size,nums_channel)
        self.stages=nn.ModuleList()
        model_dim=embed_size
        for i in range(len(block_depths)):
            # 加入SwinTransformerBlock
            stage_blocks = nn.ModuleList([
                SwinTransformerBlock(
                model_dim=model_dim,
                window_size=window_size,
                num_heads=num_heads,
                flatten_output=(j < block_depths[i]//2- 1)  # 最后一层不展开
                ) for j in range(block_depths[i]//2)
            ])

            # 加入Patch_merging
            if i < len(block_depths) - 1:  # 最后一阶段没有Patch Merging

                stage_blocks.append(patch_merging(model_dim=model_dim,merge_size=merge_size))
                model_dim *= 2  # Patch Merging通道数加倍
        
            self.stages.append(stage_blocks)

        # 分类头
        self.layer_norm=nn.LayerNorm(model_dim)
        self.header=nn.Linear(model_dim,n_classes)

    def forward(self,x):
        """
        params: x: [bs, nums_channel, height, width]
        return: x: [bs, n_classes]
        """
        x = self.patch_embedding(x)  # [bs, nums_patch, model_dim=patch_depth]
        # print(x.shape)

        for stage_blocks in self.stages:
            # print(stage_blocks)

            for stage_block in stage_blocks:
                x=stage_block(x)

                # if isinstance(stage_block,SwinTransformerBlock):
                #     print(f'transformer_block.shape:{x.shape}')
                # if isinstance(stage_block,patch_merging):
                #     print(f'patch_merging.shape:{x.shape}')

        # x.shape:[bs,nums_patch//32,model_dim*8]

        x=x.reshape(x.shape[0],-1,x.shape[-1])# 重新变换为[bs,seq_len,hidden_dim]
        print(x.shape)

        x=x.mean(dim=1)# average_pooling
        
        x = self.layer_norm(x)
        
        x=self.header(x)

        return x
    
def test():
    x=torch.randn(32,3,224,224)
    model=SwinTransformer(
        patch_size=4,
        embed_size=96,
        nums_channel=3,
        block_depths=[2,2,6,2],
        merge_size=2,
        window_size=7,
        num_heads=8,
        n_classes=10)
    # print(model)
    print(model(x).shape)
if __name__=='__main__':
    test()



