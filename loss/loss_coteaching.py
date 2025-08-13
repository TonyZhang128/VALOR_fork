import torch 
import torch.nn.functional as F
import numpy as np

def loss_coteaching(y_1, y_2, t, is_logist = False, forget_rate=0.2): # t (B, 10, 25)
    if is_logist:
        B, C = t.shape
        y_1_reshaped = y_1  # (B, 25)
        y_2_reshaped = y_2  # (B, 25)
        t_reshaped = t     # (B, 25)
    else:
        # Reshape inputs to (B*10, 25) for element-wise operations
        B, T, C = t.shape
        y_1_reshaped = y_1.reshape(-1, C)  # (B*10, 25)
        y_2_reshaped = y_2.reshape(-1, C)  # (B*10, 25)
        t_reshaped = t.reshape(-1, C)      # (B*10, 25)
    
    # Calculate element-wise loss
    if is_logist:
        loss_1 = F.binary_cross_entropy(y_1_reshaped, t_reshaped, reduction='none')
    else:
        loss_1 = F.binary_cross_entropy_with_logits(y_1_reshaped, t_reshaped, reduction='none')
    # Sum across class dimension to get per-sample loss
    loss_1 = loss_1.mean(dim=1)  # (B*10,)

    ind_1_sorted = np.argsort(loss_1.data.cpu().numpy())
    loss_1_sorted = loss_1[ind_1_sorted]

    if is_logist:
        loss_2 = F.binary_cross_entropy(y_2_reshaped, t_reshaped, reduction='none')
    else:
        loss_2 = F.binary_cross_entropy_with_logits(y_2_reshaped, t_reshaped, reduction='none')
    loss_2 = loss_2.mean(dim=1)  # (B*10,)
    
    ind_2_sorted = np.argsort(loss_2.data.cpu().numpy())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    if is_logist:
        loss_1_update = F.binary_cross_entropy(y_1_reshaped[ind_2_update], t_reshaped[ind_2_update])
        loss_2_update = F.binary_cross_entropy(y_2_reshaped[ind_1_update], t_reshaped[ind_1_update])
    else:
        loss_1_update = F.binary_cross_entropy_with_logits(y_1_reshaped[ind_2_update], t_reshaped[ind_2_update])
        loss_2_update = F.binary_cross_entropy_with_logits(y_2_reshaped[ind_1_update], t_reshaped[ind_1_update])


    if torch.isnan(loss_1_update):
        print(loss_1_update, y_1_reshaped[ind_2_update], t_reshaped[ind_2_update], len(loss_1_sorted), remember_rate)

    return loss_1_update, loss_2_update