import torch
img_mask = torch.zeros((1, 14, 14, 14, 1))  # 1 Hp Wp 1

s_slices = (slice(0, 7), slice(-7, -3), slice(-3, None))

cnt = 0
for i in s_slices:
    img_mask[:, :, :, i, :] = cnt
    cnt+=1
    print(i)
    print(img_mask)