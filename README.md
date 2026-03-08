# CLIP From Scratch

A from-scratch implementation of [CLIP (Contrastive Language-Image Pretraining)](https://arxiv.org/abs/2103.00519) using a Vision Transformer (ViT) and a 4-bit quantized causal language model with LoRA adapters.

No pretrained CLIP weights. No wrapper libraries. Just raw components wired together and trained with contrastive learning.

<img width="504" height="433" alt="Zero-shot demo — model correctly ranks the matching caption highest" src="https://github.com/user-attachments/assets/f4acc3a7-e586-48c1-945c-26f812dd29fb" />

## Architecture

```
                    ┌──────────────┐
    Image ────────► │  ViT Encoder │ ──► Linear Projection ──► L2 Norm ──┐
                    │  (768-dim)   │         (768 → 512)                  │
                    └──────────────┘                                      ├──► Cosine Similarity
                    ┌──────────────┐                                      │    + Temperature Scaling
    Text  ────────► │  Causal LM   │ ──► Linear Projection ──► L2 Norm ──┘
                    │  QLoRA 4-bit │         (1536 → 512)
                    │  (1536-dim)  │
                    └──────────────┘
```

| Component | Details |
|-----------|---------|
| **Vision Encoder** | ViT (from `transformers.ViTModel`) initialized from scratch, 768-dim CLS token output |
| **Text Encoder** | `sihab/slm-1.0` — 1.5B param causal LM, quantized to 4-bit (NF4) via BitsAndBytes, fine-tuned with LoRA (r=4, alpha=8) |
| **Projection Heads** | Two linear layers mapping each modality into a shared 512-dim embedding space |
| **Temperature** | Learnable `logit_scale` parameter (initialized to `ln(1/0.07) ≈ 2.66`, clamped at 100) |
| **Loss** | Symmetric cross-entropy over the image-text similarity matrix |

## Training

Trained on [Flickr30k](https://www.kaggle.com/datasets/eeshawn/flickr30k) image-caption pairs.

**Optimizer:** AdamW with differential learning rates
- Projection heads + logit scale: `1e-4`
- ViT encoder + LoRA adapters: `1e-5`

**Schedule:** Cosine annealing with 10% linear warmup

**Regularization:**
- Gradient clipping (max norm 1.0)
- Weight decay 0.1 (except logit scale)
- Early stopping (patience=10, min_delta=0.001)
- Data augmentation: random horizontal flip, color jitter, random grayscale

### Training Results

```
Epoch  1/20 | train: 3.5325 | val: 2.8985 | ✓ saved
Epoch  5/20 | train: 3.2878 | val: 2.7250 | ✓ saved
Epoch  9/20 | train: 3.1886 | val: 2.6800 | ✓ saved
Epoch 11/20 | train: 3.0266 | val: 2.6102 | ✓ saved  ← best
Epoch 20/20 | train: 2.9124 | val: 2.7287 | early stopping approaching
```

Best validation loss: **2.6102**

## Usage

### Setup

```bash
pip install torch transformers peft bitsandbytes torchvision pillow
```

### Load the Model

```python
from model import VitEncoder, TextEncoder, ClipTransformer

# Initialize encoders
vision_encoder = VitEncoder()
text_encoder = TextEncoder()

# Build CLIP
clip_model = ClipTransformer(
    text_encoder=text_encoder,
    vision_encoder=vision_encoder,
    text_dimension=1536,
    vision_dimension=768,
    projection_dim=512,
)
```

### Load Trained Weights

```python
import torch
from peft import PeftModel

# Load projections + logit scale
ckpt = torch.load("best_clip/projections.pt", map_location="cuda:0")
clip_model.vision_projection.load_state_dict(ckpt["vision_projection"])
clip_model.text_projection.load_state_dict(ckpt["text_projection"])
clip_model.logit_scale.data = ckpt["logit_scale"].to("cuda:0")

# Load ViT weights
clip_model.vision_encoder.load_state_dict(
    torch.load("best_clip/vision_encoder.pt", map_location="cuda:0")
)

# Load LoRA adapters
clip_model.text_encoder.lora_model = PeftModel.from_pretrained(
    clip_model.text_encoder.model, "best_clip/lora_weights"
)
```

### Zero-Shot Image Classification

```python
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

clip_model.eval()

img = transform(Image.open("dog.jpg").convert("RGB")).unsqueeze(0).to("cuda:0")

candidates = [
    "A golden retriever catches a tennis ball in the snow.",
    "A cat sleeping on a couch.",
    "A group of people dancing at a party.",
    "A dog running through a snowy field.",
    "A man riding a bicycle on a road.",
]

with torch.no_grad():
    img_emb = clip_model.encode_image(img)
    txt_emb = clip_model.encode_text(candidates)
    sims = (img_emb @ txt_emb.T).squeeze(0)
    probs = sims.softmax(dim=-1)

for caption, prob in zip(candidates, probs):
    print(f"[{prob:.2%}] {caption}")
```

### Image-Text Retrieval

```python
# Encode a pool of images and captions
img_embs = clip_model.encode_image(image_batch)   # [N, 512]
txt_embs = clip_model.encode_text(caption_list)    # [N, 512]

# Full similarity matrix
sims = img_embs @ txt_embs.T  # [N, N]

# Image → Text: for each image, find the best matching caption
i2t_matches = sims.argmax(dim=1)

# Text → Image: for each caption, find the best matching image
t2i_matches = sims.argmax(dim=0)
```

## Training Your Own

```python
from trainer import CLIPTrainer

trainer = CLIPTrainer(
    train_dl=train_dl,
    val_dl=val_dl,
    clip_model=clip_model,
    optimizer=optimizer,
    scheduler=scheduler,
    patience=10,
    min_delta=0.001,
)

train_losses, val_losses = trainer.fit(epochs=20)
```

## Key Takeaways

**CLIP doesn't generate text.** It ranks candidates by similarity. Given an image and a list of captions, it tells you which caption best matches — but it cannot produce new descriptions on its own.

**Saving QLoRA models requires care.** You can't `torch.save()` the entire model because the 4-bit quantized base doesn't serialize cleanly. LoRA adapters must be saved separately via `save_pretrained()` and reloaded on top of a freshly instantiated base model.

**Temperature matters.** The learned `logit_scale` controls how sharply the model distinguishes between candidates. Too high → can't separate hard negatives. Too low → vanishing gradients.

## What's Next

Adding a projection bridge from the vision encoder into the causal LM's embedding space to enable **image captioning** — essentially building a minimal [LLaVA](https://arxiv.org/abs/2304.08485) on top of this CLIP backbone.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- PEFT
- BitsAndBytes
- torchvision
- Pillow

## License

MIT
