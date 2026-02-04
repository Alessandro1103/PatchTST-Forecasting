import torch
import torch.nn as nn

# Creo un patcher che sia flessibile, dove posso scegliere se fare overlapping tra i patch che ho creato (PatchTST) oppure evitarlo (PatchTST self-supervising). 

class Patcher(nn.Module):
  
    def __init__(self, patch_len, stride):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
  
    def forward(self, x, mask_ratio=0.0):

        # input = batch x channels x length

        batch, channels, length = x.shape
            
        padding = x[:, :, -1].unsqueeze(dim=2).repeat(1, 1, self.stride) # batch x channels x stride

        x_padded = torch.cat([x, padding], 2)

        n_patches = ((length - self.patch_len) // self.stride) + 2

        patches = torch.zeros((batch, channels, n_patches, self.patch_len), device=x.device)

        index = 0
        for i in range(n_patches):
            patches[:, :, i, :] = x_padded[:, :, index : index + self.patch_len]
            index += self.stride 

        if mask_ratio > 0:
            num_masked = int(mask_ratio * n_patches)
            
            noise = torch.rand(batch, channels, n_patches, device=x.device)
            
            ids_shuffle = torch.argsort(noise, dim=2)  
            ids_restore = torch.argsort(ids_shuffle, dim=2)
            
            mask = torch.zeros(batch, channels, n_patches, device=x.device)
            mask[:, :, :num_masked] = 1.0 
            
            mask = torch.gather(mask, dim=2, index=ids_restore)
            
            masked_patches = patches * (1 - mask.unsqueeze(-1))
            
            return masked_patches, mask
        
        # No mask case
        else:
            return patches, None

if __name__ == "__main__":
    x = torch.rand(3, 4, 336) # [Batch, Channel, Length]
    print("Input shape:\n", x.shape)
    
    patcher = Patcher(patch_len=16, stride=8)
    
    # Test without mask
    out, mask = patcher(x)
    print(f"Output with no_mask: {out.shape}, Mask is None: {mask is None}")

    # Test with mask (40%)
    out_masked, mask_gen = patcher(x, mask_ratio=0.4)
    print(f"Output with mask: {out_masked.shape}")
