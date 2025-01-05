import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class WindowsMultiheadAttention(nn.Module):# 基于窗口的多头注意力
    def __init__(self,model_dim=96, num_heads=8, window_size=7,dropout=0.1):
        super(WindowsMultiheadAttention, self).__init__()
        self.window_size=window_size
        self.attention=nn.MultiheadAttention(model_dim, num_heads,batch_first=True,dropout=dropout)

    def forward(self, x):
        """
        1.将patch展开为图片
        2.将图片划分为窗口
        3.将窗口中的patch分割，进行多头注意力计算
        """
        # x.shape: [batch_size, nums_patch, patch_depth]
        bs,nums_patch,patch_depth=x.shape

        patch_size=int(math.sqrt(nums_patch))# h,w

        nums_patch_in_window=self.window_size*self.window_size
        
        x=x.transpose(1,2).reshape(bs,patch_depth,patch_size,patch_size)# 转换为便于unfold接受的维度 [bs, patch_depth, h, w]

        # 获取窗口数和窗口维度
        nums_window=nums_patch//nums_patch_in_window

        window_depth=patch_depth*nums_patch_in_window

        # 进行窗口化
        x = F.unfold(x,kernel_size=self.window_size,stride=self.window_size)# [bs, window_depth, nums_window]

        x=x.reshape(bs,window_depth,nums_window)
        
        
        x=x.transpose(1,2)# [bs, nums_window, window_depth]
        
        x=x.reshape(bs*nums_window,nums_patch_in_window,patch_depth)# [bs*nums_window, h*w, patch_depth]

        x,attention_weight=self.attention(x,x,x)# [bs*nums_window, h*w, patch_depth]

        x=x.reshape(bs,nums_window,nums_patch_in_window,patch_depth)# [bs, nums_window, h*w, patch_depth]
        # print(f'nums_window={nums_window},nums_patch_in_window={nums_patch_in_window},patch_depth={patch_depth}')

        return x,attention_weight
if __name__=='__main__':
    x=torch.randn(32,3136,96) # 32,224*224/16,96

    model=WindowsMultiheadAttention()

    print(model(x)[0].shape)