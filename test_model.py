import torch
# from safetensors.torch import load_file
from dinov3.models.vision_transformer import vit_small

# --- Build ViT-S/16 model ---
IMG_SIZE = 224
PATCH_SZ = 16
model = vit_small(patch_size=PATCH_SZ, img_size=IMG_SIZE)

# --- Load safetensors weights ---
ckpt_path = "/mnt/isilon/aali/temporal_correlation/dinov3_Temporal/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# --- Dummy video batch (B,T,C,H,W) ---
B, T, C, H, W = 1, 2, 3, IMG_SIZE, IMG_SIZE
video = torch.rand(B, T, C, H, W, device=device)
x = video.view(B*T, C, H, W)

# --- Forward pass ---
with torch.no_grad():
    out = model.forward_features(x)  # dict

print("Keys:", out.keys())

cls     = out["x_norm_clstoken"].view(B, T, -1)               # (B,T,D)
patches = out["x_norm_patchtokens"].view(B, T, -1, cls.size(-1))  # (B,T,P,D)

print("CLS:", cls.shape)
print("patches:", patches.shape)