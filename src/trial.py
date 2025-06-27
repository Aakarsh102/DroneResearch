import torch

def create_autoregressive_prefix_mask(
    total_len: int = 11,
    prefix_end_idx: int = 5,           # prefix runs from 0..5
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    prefix_len = prefix_end_idx + 1
    target_len = total_len - prefix_len

    # start with everything masked
    mask = torch.full((total_len, total_len), float('-inf'), device=device)

    # 1) prefix → prefix: full attention
    mask[:prefix_len, :prefix_len] = 0.0

    # 2) target → prefix: full attention
    mask[prefix_len:, :prefix_len] = 0.0

    # 3) target → target: causal (j > i blocked)
    causal = torch.triu(torch.ones(target_len, target_len, device=device), diagonal=1)
    causal = causal.masked_fill(causal == 1, float('-inf'))
    mask[prefix_len:, prefix_len:] = causal

    return mask

# Example: build your 11×11 mask
mask11 = create_autoregressive_prefix_mask(11, 5)
print(mask11)

# print(create_prefix_causal_mask(5,5))
# print(create_prefix_causal_mask(5,6))