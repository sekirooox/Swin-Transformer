import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,model_dim=96, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.model_dim=model_dim
        self.nums_head=num_heads
        self.kqv=nn.Linear(model_dim,model_dim*3)
        self.output_layer=nn.Linear(model_dim,model_dim)
    def forward(self,x,mask=None):
        """
        params: x:[bs,nums_patch,patch_depth]
        params:mask:[bs,nums_patch,nums_patch]
        """
        bs,nums_patch,model_dim=x.shape
        nums_head=self.nums_head
        head_dim=model_dim//nums_head

        # 获取k,v,q
        kqv=self.kqv(x)
        k,q,v=kqv.chunk(3,dim=-1)

        # 获取多头注意力
        k=k.reshape(bs,nums_patch,nums_head,head_dim).transpose(1,2)
        k=k.reshape(bs*nums_head,nums_patch,head_dim)

        q=q.reshape(bs,nums_patch,nums_head,head_dim).transpose(1,2)
        q=q.reshape(bs*nums_head,nums_patch,head_dim)

        v=v.reshape(bs,nums_patch,nums_head,head_dim).transpose(1,2)
        v=v.reshape(bs*nums_head,nums_patch,head_dim)

        # 计算注意力
        qk=torch.bmm(q,k.transpose(1,2))/math.sqrt(head_dim) # [bs*nums_head,nums_patch,nums_patch]
        if mask is not None:
            mask=mask.tile((nums_head,1,1))# 复制mask,满足多头注意力的计算 mask:[bs*nums_head,nums_patch,nums_patch]
            qk+=mask
        attention_score=F.softmax(qk,dim=-1)
        output=torch.bmm(attention_score,v)

        # 获取输出
        output=output.reshape(bs,nums_head,nums_patch,head_dim).transpose(1,2)
        output=output.reshape(bs,nums_patch,model_dim)
        output=self.output_layer(output)


        return output,attention_score # [bs,nums_patch,model_dim],[bs*nums_head,nums_patch,nums_patch]
def window_to_image(window):
    # window:[bs, nums_window, h*w, patch_depth]
    bs,nums_window,nums_patch_in_window,patch_depth=window.shape

    # 获取批量宽高
    h_patch=int(math.sqrt(nums_patch_in_window))
    w_patch=h_patch

    # 获取每个窗口的宽高
    h_window=int(math.sqrt(nums_window))
    w_window=h_window

    # 重新调整窗口形状
    window=window.reshape(bs,h_window,w_window,h_patch,w_patch,patch_depth).transpose(2,3)# [bs,h_window,h_patch,w_window,w_patch,patch_depth]

    window=window.reshape(bs,h_window*h_patch,w_window*w_patch,patch_depth)# [bs,h,w,patch_depth]

    return window 
def test_window_to_image():
    window=torch.randn(32,196,16,96)
    image=window_to_image(window)
    print(image.shape)
def cycle_shift(window,window_size,shift_size,require_mask=False):
    """
    1.窗口转图片
    2.图片进行循环移位
    3.图片转窗口
    4.生成掩码
    :param window: 输入张量，形状为 [bs, nums_window, h*w, patch_depth]
    :param window_size: 窗口的边长
    :param shift_size: 移位的大小
    :return: window:[bs,nums_window,h*w,patch_depth]
    """
    bs,nums_window,nums_patch_in_window,patch_depth=window.shape

    # 转换为图片
    image=window_to_image(window)# [bs,h,w,patch_depth]

    
    # 循环移位图像
    rolled_image=torch.roll(image, shifts=(shift_size,shift_size), dims=(1, 2))# [bs,h,w,patch_depth]
    bs,h,w,patch_depth=rolled_image.shape# h,w是以批次为单位的宽高

    # 获取批量宽高
    h_patch=int(math.sqrt(nums_patch_in_window))
    w_patch=h_patch

    # 获取每个窗口的宽高
    h_window=int(math.sqrt(nums_window))
    w_window=h_window

    # 转换为窗口
    rolled_window=rolled_image.reshape(bs,h_window,h_patch,w_window,w_patch,patch_depth)
    rolled_window=rolled_image.transpose(2,3)# [bs,h_window,w_window,h_patch,w_patch,patch_depth]
    rolled_window=rolled_image.reshape(bs,h_window*w_window,h_patch*w_patch,patch_depth)# [bs,nums_window,h*w,patch_depth]

    # # 生成循环移位的掩码
    if require_mask:
        additive_mask=generate_mask(bs,h,w,window_size)
    else:
        additive_mask=None
    
    return rolled_window,additive_mask# [bs,nums_window,h*w,patch_depth],[bs*nums_window,nums_patch_in_window,nums_patch_in_window]
def generate_mask(bs,image_h,image_w,window_size):
    """
    params:image_h:patch的高度
    params:window_size:窗口长度
    """
    # 根据窗口边长生成标记
    index_matrix = torch.zeros(image_h, image_w)
    for i in range(image_h):
        for j in range(image_w):
            row_times = i // window_size
            col_times = j // window_size
            index_matrix[i, j] = row_times * (image_w // window_size) + col_times + 1


    # 循环移位半个window长度
    rolled_index_matrix = torch.roll(index_matrix, shifts=(window_size // 2, window_size // 2), dims=(0, 1))# 论文：移动窗口大小的一半

    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)# [bs=1,c=1,h,w]


    # 生成窗口索引:由于通道数为1，所以nums_patch_in_window*1=patch_depth
    window = F.unfold(rolled_index_matrix, kernel_size=window_size, stride=window_size).transpose(-1, -2)# [bs=1,nums_window,nums_patch_in_window]
    

    # 扩展bs维度
    window = window.tile(bs, 1, 1)

    bs,nums_window,nums_patch_in_window=window.shape


    # 构建同类关系矩阵:简单理解为每一个窗口内进行mask
    c1 = window.unsqueeze(-1)  
    c2 = ((c1 - c1.transpose(-1, -2))!=0).to(torch.float32)# 结果不等于0：同一个窗口，等于0：不同窗口 
    
    additive_mask=c2*(-1e6)# 加性掩码

    additive_mask = additive_mask.reshape(bs * nums_window, nums_patch_in_window, nums_patch_in_window)
    
    return additive_mask# [bs*nums_window,nums_patch_in_window,nums_patch_in_window]
class ShiftWindowsMultiheadAttention(nn.Module):
    def __init__(self,model_dim,window_size=7,num_heads=8):
        super(ShiftWindowsMultiheadAttention, self).__init__()
        self.num_heads=num_heads
        self.model_dim=model_dim
        self.window_size=window_size
        self.attention=MultiHeadAttention(model_dim,num_heads)


    def forward(self,window):
        """
        params x: [bs, nums_window, h*w, patch_depth]
        return x: [bs,nums_window,h*w,patch_depth] or [bs,nums_patch,patch_depth]
        """
        bs,nums_window,nums_patch_in_window,patch_depth=window.shape

        window_size=self.window_size
        
        rolled_window,additive_mask=cycle_shift(window,window_size,window_size//2,require_mask=True)# [bs, nums_window, h*w, patch_depth]

        rolled_window=rolled_window.reshape(bs*nums_window,nums_patch_in_window,patch_depth)# [bs*nums_window, h*w, patch_depth]

        output,attention_score=self.attention(rolled_window,additive_mask)# [bs*nums_window, h*w, patch_depth]

        output=output.reshape(bs,nums_window,nums_patch_in_window,patch_depth)

        output,_=cycle_shift(output,window_size,-window_size//2,require_mask=False)# [bs, nums_window, h*w, patch_depth]
        
        return output,attention_score
def test_mask():
    bs=1
    image_h=56 
    image_w=56
    window_size=7
    additive_mask=generate_mask(bs,image_h,image_w,window_size)
    print(additive_mask[0,:21,:21])
def test_cycle_shift():
    bs=1
    nums_window=4
    window_size=2
    nums_patch_in_window=4
    patch_depth=96
    window=torch.randn(bs,nums_window,nums_patch_in_window,patch_depth)
    print(window.shape)
    window,mask=cycle_shift(window,window_size=2,shift_size=1,require_mask=True)

    print(window.shape)
    print(mask.shape)
    print(window)
    print(mask)
def test_shift_windows_attention():
    bs = 32
    nums_window = 64
    nums_patch_in_window = 49  
    patch_depth = 96
    window_size = 7

    # 随机输入窗口数据
    window = torch.randn(bs, nums_window, nums_patch_in_window, patch_depth)

    # 定义 ShiftWindowsMultiheadAttention
    attention = ShiftWindowsMultiheadAttention(model_dim=96, window_size=window_size, num_heads=8)

    # 前向计算
    output, attention_score = attention(window)

    print("输出形状:", output.shape)  # [bs, nums_window, nums_patch_in_window, patch_depth]
    print("注意力得分形状:", attention_score.shape)  # [bs * nums_window * num_heads, nums_patch_in_window, nums_patch_in_window]

if __name__ == '__main__':
    # test_window_to_image()
    # test_mask()
    # test_cycle_shift()
    test_shift_windows_attention()