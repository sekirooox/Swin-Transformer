import torch
import torch.nn as nn
import torch.nn.functional as F
class patch_embedding(nn.Module):
    def __init__(self,patch_size=4,embed_size=96,nums_channel=3):

        super(patch_embedding, self).__init__()

        self.patch_size = patch_size
        self.patch_depth=patch_size*patch_size*nums_channel
        self.linear_embedding=nn.Linear(self.patch_depth,embed_size)

        
    def forward(self,x):
        # x: [batch_size,nums_channel,height,width]
        patch_size=self.patch_size
        x=F.unfold(x,kernel_size=patch_size,stride=patch_size).transpose(-1,-2)

        x=self.linear_embedding(x)

        return x
if __name__ == '__main__':
    example=torch.zeros([2,3,224,224])
    model=patch_embedding()
    print(model(example).shape)
         