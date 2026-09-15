import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# PENTROVA FINAL RESEARCH IMPLEMENTATION
# ============================================================
# Input: 1080x1080 RGB
# FLASH: cheap 224x224 global analysis -> variable-size crops
# Eye: frozen ResNet-18 Stem -> Layer1 -> Layer2 -> Layer3
#      -> adaptive 8x8 spatial feature matrix (256 channels)
# PENTROVA x4: LN -> bidirectional spatial QKV -> positional
# relational context -> cosine relational attention -> residual
# -> heterogeneous KAN-like nonlinear wires -> residual
# Feedback can request extra regions from the original image.
#
# Important: hard crop selection is inherently non-differentiable.
# This implementation therefore uses a straight-through soft gate for
# controller learning while preserving hard selected regions in forward.
# Region coordinates are kept in ORIGINAL-image normalized coordinates.
# ============================================================


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Region:
    x_center: float
    y_center: float
    width: float
    height: float
    score: float
    region_id: int

    def as_tensor(self, device):
        base = torch.tensor(
            [self.x_center, self.y_center, self.width, self.height],
            dtype=torch.float32,
            device=device,
        )
        score = self.score if torch.is_tensor(self.score) else torch.tensor(float(self.score), dtype=torch.float32, device=device)
        return torch.cat([base, score.reshape(1).to(device=device, dtype=torch.float32)], dim=0)


# ============================================================
# FLASH
# ============================================================

class TinyCNNController(nn.Module):
    """Cheap global controller. It scores a fixed multi-scale proposal bank."""

    def __init__(self, hidden: int = 64, num_proposals: int = 36):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, num_proposals),
        )

    def forward(self, x):
        z = self.features(x)
        logits = self.score(z)
        return logits


class FlashProposalBank:
    """Multi-scale regions in original-image normalized coordinates."""

    def __init__(self, grid=(3, 3), scales=(0.28, 0.42, 0.60, 0.78)):
        self.proposals: List[Region] = []
        rid = 0
        gx, gy = grid
        # centers are arranged so every box remains inside [0,1].
        for scale in scales:
            half = scale / 2.0
            xs = torch.linspace(half, 1.0 - half, gx).tolist()
            ys = torch.linspace(half, 1.0 - half, gy).tolist()
            for y in ys:
                for x in xs:
                    self.proposals.append(Region(x, y, scale, scale, 0.0, rid))
                    rid += 1
        # Add two full-width/height asymmetric proposals for context.
        self.proposals.extend([
            Region(0.50, 0.50, 1.00, 0.55, 0.0, rid),
            Region(0.50, 0.50, 0.55, 1.00, 0.0, rid + 1),
        ])

    def __len__(self):
        return len(self.proposals)

    def tensors(self, device):
        return torch.tensor(
            [[p.x_center, p.y_center, p.width, p.height] for p in self.proposals],
            dtype=torch.float32,
            device=device,
        )


class FLASH(nn.Module):
    """Foveated Learnable Adaptive Sampling / Object Extraction."""

    def __init__(self, num_proposals=38, max_initial_regions=6, min_score=0.18):
        super().__init__()
        self.controller = TinyCNNController(hidden=64, num_proposals=num_proposals)
        self.max_initial_regions = max_initial_regions
        self.min_score = min_score
        self.register_buffer("proposal_boxes", FlashProposalBank().tensors(torch.device("cpu")), persistent=False)

    def proposal_boxes_device(self, device):
        return self.proposal_boxes.to(device=device)

    @staticmethod
    def _select_indices(scores, max_regions, min_score):
        # Always select at least one proposal.
        order = torch.argsort(scores, descending=True)
        selected = [int(i) for i in order[:max_regions].tolist() if scores[i].detach().item() >= min_score]
        if not selected:
            selected = [int(order[0].item())]
        return selected

    def forward(self, image, global_size=224):
        # image: B,C,H,W. Cropping later is performed on ORIGINAL image.
        global_img = F.interpolate(image, size=(global_size, global_size), mode="bilinear", align_corners=False)
        logits = self.controller(global_img)
        scores = torch.sigmoid(logits)
        boxes = self.proposal_boxes_device(image.device)
        all_regions = []
        all_weights = []
        for b in range(image.shape[0]):
            idx = self._select_indices(scores[b], self.max_initial_regions, self.min_score)
            regs = []
            weights = []
            for i in idx:
                box = boxes[i]
                regs.append(Region(float(box[0]), float(box[1]), float(box[2]), float(box[3]), scores[b, i], i))
                weights.append(scores[b, i])
            all_regions.append(regs)
            all_weights.append(torch.stack(weights))
        return all_regions, all_weights, scores, boxes

    @staticmethod
    def crop_original(image, region: Region):
        """Crop directly from the original image; no forced resize of the object."""
        _, h, w = image.shape
        x0 = max(0, min(w - 1, int(round((region.x_center - region.width / 2) * w))))
        x1 = max(x0 + 1, min(w, int(round((region.x_center + region.width / 2) * w))))
        y0 = max(0, min(h - 1, int(round((region.y_center - region.height / 2) * h))))
        y1 = max(y0 + 1, min(h, int(round((region.y_center + region.height / 2) * h))))
        return image[:, y0:y1, x0:x1], (x0, y0, x1, y1)

    @staticmethod
    def pad_crops(crops: List[torch.Tensor]):
        """Batch variable-size crops by aspect-ratio-preserving padding."""
        if not crops:
            raise ValueError("FLASH produced no crops.")
        max_h = max(c.shape[-2] for c in crops)
        max_w = max(c.shape[-1] for c in crops)
        batch = []
        masks = []
        for c in crops:
            _, h, w = c.shape
            # Replicate padding avoids a hard black border.
            padded = F.pad(c.unsqueeze(0), (0, max_w - w, 0, max_h - h), mode="replicate").squeeze(0)
            mask = torch.zeros((1, max_h, max_w), device=c.device, dtype=c.dtype)
            mask[:, :h, :w] = 1.0
            batch.append(padded)
            masks.append(mask)
        return torch.stack(batch, dim=0), torch.stack(masks, dim=0)


# ============================================================
# THE EYE — frozen ResNet-18 through Layer 3
# ============================================================

class FrozenResNet18Eye(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                net = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception as exc:
                raise RuntimeError(
                    "ImageNet ResNet-18 weights could not be loaded. "
                    "Run once with internet access so torchvision can download the weights, "
                    "or set pretrained=False."
                ) from exc
        else:
            net = resnet18(weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Always keep frozen Eye in eval mode so BatchNorm statistics do not change.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class CropFeatureEncoder(nn.Module):
    """Eye + 256->96 + spatial adaptive 8x8 representation."""

    def __init__(self, pretrained_eye=True):
        super().__init__()
        self.eye = FrozenResNet18Eye(pretrained=pretrained_eye)
        self.linear = nn.Linear(256, 96)
        self.norm = nn.LayerNorm(96)

    def forward(self, crops):
        with torch.no_grad():
            feat = self.eye(crops)  # B,256,Hf,Wf
        feat = F.adaptive_avg_pool2d(feat, (8, 8))
        feat = feat.permute(0, 2, 3, 1).contiguous()  # B,8,8,256
        feat = self.linear(feat)  # B,8,8,96
        feat = self.norm(feat)
        return feat.view(feat.shape[0], 64, 96)


# ============================================================
# PENTROVA RELATIONAL CORE
# ============================================================

class PositionalRelationalContext(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 48),
            nn.SiLU(),
            nn.Linear(48, dim),
        )

    def forward(self, tokens, region_meta):
        # tokens: B,T,D; region_meta: B,T,5
        return tokens + self.mlp(region_meta)


class BidirectionalSpatialQKV(nn.Module):
    def __init__(self, dim=96, heads=4, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        out = attn.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class CosineRelationalAttention(nn.Module):
    def __init__(self, dim=96, heads=4, temperature=0.07):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, x):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        temp = self.log_temperature.exp().clamp(0.01, 10.0)
        logits = torch.matmul(q, k.transpose(-2, -1)) / temp
        weights = torch.softmax(logits, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class HeterogeneousKANLayer(nn.Module):
    """KAN-like nonlinear wires with heterogeneous edge basis functions."""

    def __init__(self, dim=96, knots=8, basis_width=0.35):
        super().__init__()
        self.dim = dim
        self.knots = knots
        self.base_weight = nn.Parameter(torch.empty(dim, dim))
        self.base_bias = nn.Parameter(torch.zeros(dim))
        self.coeff = nn.Parameter(torch.zeros(dim, dim, knots))
        self.knot_pos = nn.Parameter(torch.linspace(-1.0, 1.0, knots), requires_grad=False)
        self.log_width = nn.Parameter(torch.log(torch.full((knots,), basis_width)))
        self.activation_id = nn.Parameter(torch.arange(dim, dtype=torch.float32), requires_grad=False)
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.coeff, std=0.01)

    def forward(self, x):
        # Normalize each input wire into a stable range for the spline/RBF basis.
        z = torch.tanh(x)
        # [B,T,D,K]
        diff = z.unsqueeze(-1) - self.knot_pos.view(1, 1, 1, self.knots)
        width = self.log_width.exp().view(1, 1, 1, self.knots).clamp_min(1e-3)
        basis = torch.exp(-0.5 * (diff / width) ** 2)
        # Aggregate each input dimension's nonlinear edge function into each output.
        nonlinear = torch.einsum("btdk,odk->bto", basis, self.coeff)
        linear = torch.einsum("btd,do->bto", x, self.base_weight) + self.base_bias
        # Heterogeneous neuron nonlinearities by output-neuron group.
        idx = torch.arange(self.dim, device=x.device)
        group = idx % 5
        funcs = torch.stack([
            torch.tanh(linear),
            F.silu(linear),
            torch.sin(linear),
            F.softplus(linear) - math.log(2.0),
            linear + 0.15 * linear.pow(2),
        ], dim=-1)  # B,T,D,5
        hetero = funcs.gather(-1, group.view(1, 1, self.dim, 1).expand(x.shape[0], x.shape[1], -1, 1)).squeeze(-1)
        return hetero + nonlinear


class FlashFeedback(nn.Module):
    """Scores not-yet-used proposals using current relational state."""

    def __init__(self, dim=96, meta_dim=5):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim + meta_dim, dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, state, proposal_embeddings, proposal_meta, used_mask):
        # state B,D; proposal_embeddings B,N,D; meta B,N,5
        q = F.normalize(self.query(state), dim=-1)
        p = torch.cat([proposal_embeddings, proposal_meta], dim=-1)
        k = F.normalize(self.key(p), dim=-1)
        score = torch.sum(q.unsqueeze(1) * k, dim=-1) / self.temperature.clamp_min(0.05)
        score = score.masked_fill(used_mask, -1e9)
        return score


class PentrovaBlock(nn.Module):
    def __init__(self, dim=96, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_qkv = BidirectionalSpatialQKV(dim, heads)
        self.positional = PositionalRelationalContext(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cosine = CosineRelationalAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.kan = HeterogeneousKANLayer(dim)

    def forward(self, x, meta):
        y = self.norm1(x)
        y = self.spatial_qkv(y)
        y = self.positional(y, meta)
        x = x + y
        y = self.cosine(self.norm2(x))
        x = x + y
        y = self.kan(self.norm3(x))
        x = x + y
        return x


# ============================================================
# READOUT
# ============================================================

class ObjectQueryReadout(nn.Module):
    def __init__(self, dim=96, num_queries=8, num_classes=101, heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, num_classes)
        self.object_head = nn.Linear(dim, 5)  # confidence + x,y,w,h in normalized coords

    def forward(self, tokens, valid_mask=None):
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        key_padding_mask = None if valid_mask is None else ~valid_mask
        q2, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm(q + q2)
        pooled = q.mean(dim=1)
        logits = self.cls(pooled)
        object_queries = self.object_head(q)
        return logits, object_queries, q


# ============================================================
# COMPLETE MODEL
# ============================================================

class PENTROVA(nn.Module):
    def __init__(
        self,
        num_classes=101,
        pretrained_eye=True,
        max_initial_regions=6,
        feedback_regions=1,
        num_blocks=4,
    ):
        super().__init__()
        self.flash = FLASH(max_initial_regions=max_initial_regions)
        self.eye = CropFeatureEncoder(pretrained_eye=pretrained_eye)
        self.blocks = nn.ModuleList([PentrovaBlock(96, 4) for _ in range(num_blocks)])
        self.feedback = FlashFeedback(96, 5)
        self.readout = ObjectQueryReadout(96, num_queries=8, num_classes=num_classes)
        self.feedback_regions = feedback_regions

    def _extract_selected(self, image, regions):
        crops = []
        metas = []
        for r in regions:
            crop, _ = self.flash.crop_original(image, r)
            crops.append(crop)
            # Per-region metadata is normalized and duplicated across its 64 spatial tokens.
            meta = r.as_tensor(image.device)
            metas.append(meta)
        batch, _ = self.flash.pad_crops(crops)
        # Normalize only the image tensor fed to the pretrained Eye.
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        feats = self.eye(batch)
        # Preserve hard region selection while keeping the selected FLASH scores
        # in the computation graph. This gives the controller a straight-through
        # learning signal from the classification objective.
        scores = torch.stack([r.as_tensor(image.device)[4] for r in regions]).to(feats.dtype)
        feats = feats * scores[:, None, None]
        meta = torch.stack(metas, dim=0).to(feats.dtype)
        meta = meta[:, None, :].expand(-1, 64, -1).reshape(-1, 5)
        feats = feats.reshape(-1, 96)
        return feats, meta

    @staticmethod
    def _tokens_to_regions(features, meta, num_regions):
        # Already ordered region-by-region, 64 tokens per region.
        return features, meta

    def forward(self, image, return_debug=False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image tensor shaped [B,3,H,W].")

        region_lists, _, flash_scores, proposal_boxes = self.flash(image)
        final_tokens = []
        final_meta = []
        debug = []

        for b in range(image.shape[0]):
            regions = region_lists[b]
            used_ids = {r.region_id for r in regions}
            tokens, meta = self._extract_selected(image[b], regions)
            # tokens are N*64 x 96
            used_mask = torch.zeros(1, proposal_boxes.shape[0], dtype=torch.bool, device=image.device)
            for rid in used_ids:
                used_mask[0, rid] = True

            # Build cheap proposal embeddings from FLASH scores + coordinates.
            controller_scores = torch.sigmoid(self.flash.controller(
                F.interpolate(image[b:b+1], size=(224, 224), mode="bilinear", align_corners=False)
            ))[0]
            proposal_meta = torch.cat([
                proposal_boxes,
                controller_scores.unsqueeze(-1),
            ], dim=-1)
            # Project metadata to the feedback key input space through a small deterministic embedding.
            prop_emb = F.pad(proposal_meta, (0, 91))[:, :96].unsqueeze(0)
            proposal_meta_batched = proposal_meta.unsqueeze(0)

            for block_id, block in enumerate(self.blocks):
                # Pentrova block processes every currently selected region bidirectionally.
                x = tokens.unsqueeze(0)
                m = meta.unsqueeze(0)
                x = block(x, m)
                tokens = x.squeeze(0)

                # Feedback: after each block, request at most one extra useful region.
                if block_id < len(self.blocks) - 1 and self.feedback_regions > 0:
                    state = tokens.mean(dim=0, keepdim=True).unsqueeze(0)
                    fscore = self.feedback(state.squeeze(1), prop_emb, proposal_meta_batched, used_mask)
                    best = int(torch.argmax(fscore[0]).item())
                    if fscore[0, best].detach().item() > -1e8:
                        box = proposal_boxes[best]
                        new_region = Region(float(box[0]), float(box[1]), float(box[2]), float(box[3]), flash_scores[b, best], best)
                        # Avoid pathological duplicate selection.
                        used_mask[0, best] = True
                        nt, nm = self._extract_selected(image[b], [new_region])
                        tokens = torch.cat([tokens, nt], dim=0)
                        meta = torch.cat([meta, nm], dim=0)
                        regions.append(new_region)

            final_tokens.append(tokens)
            final_meta.append(meta)
            debug.append(regions)

        # Samples may have different region counts. Pad token sequences for readout.
        max_t = max(t.shape[0] for t in final_tokens)
        bsz = len(final_tokens)
        token_batch = image.new_zeros((bsz, max_t, 96))
        meta_batch = image.new_zeros((bsz, max_t, 5))
        valid = torch.zeros((bsz, max_t), dtype=torch.bool, device=image.device)
        for i, (t, m) in enumerate(zip(final_tokens, final_meta)):
            token_batch[i, :t.shape[0]] = t
            meta_batch[i, :m.shape[0]] = m
            valid[i, :t.shape[0]] = True

        # Mask padding before readout by zeroing it; attention then cannot be perfectly masked,
        # so use a large negative additive effect through the token norm-free zero representation.
        logits, object_queries, query_states = self.readout(token_batch, valid_mask=valid)

        if return_debug:
            return logits, {
                "object_queries": object_queries,
                "query_states": query_states,
                "flash_scores": flash_scores,
                "proposal_boxes": proposal_boxes,
                "regions": debug,
                "valid_tokens": valid,
                "token_batch": token_batch,
                "meta_batch": meta_batch,
            }
        return logits


# ============================================================
# DATA / TRAINING
# ============================================================

class Food101Data:
    def __init__(self, root, image_size=1080):
        # Training transform keeps the 1080x1080 architecture input.
        self.train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.test_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.train = datasets.Food101(root=root, split="train", transform=self.train_tf, download=True)
        self.test = datasets.Food101(root=root, split="test", transform=self.test_tf, download=True)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, amp=True):
    model.train()
    model.eye.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_amp = amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach()) * labels.size(0)
        total_correct += int((logits.argmax(1) == labels).sum())
        total += labels.size(0)

        if step % 20 == 0:
            print(f"Epoch {epoch} | step {step}/{len(loader)} | loss {loss.item():.4f} | acc {total_correct/total:.4f}")

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    model.eye.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    use_amp = amp and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss) * labels.size(0)
        total_correct += int((logits.argmax(1) == labels).sum())
        total += labels.size(0)
    return total_loss / total, total_correct / total


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all(model):
    return sum(p.numel() for p in model.parameters())


def build_model(args):
    return PENTROVA(
        num_classes=args.num_classes,
        pretrained_eye=not args.random_eye,
        max_initial_regions=args.max_regions,
        feedback_regions=args.feedback_regions,
        num_blocks=4,
    )


def main():
    parser = argparse.ArgumentParser(description="PENTROVA final architecture")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--max-regions", type=int, default=6)
    parser.add_argument("--feedback-regions", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-eye", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="pentrova_final.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data = Food101Data(args.data, image_size=args.image_size)
    train_loader = DataLoader(
        data.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        data.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = build_model(args).to(device)
    print("ALL PARAMS:", f"{count_all(model):,}")
    print("TRAINABLE:", f"{count_trainable(model):,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, amp=not args.no_amp)
        te_loss, te_acc = evaluate(model, test_loader, device, amp=not args.no_amp)
        print(f"EPOCH {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
        if te_acc > best:
            best = te_acc
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "test_acc": te_acc,
                "args": vars(args),
            }, args.checkpoint)
            print("Saved best checkpoint:", args.checkpoint)

    print("BEST TEST ACC:", best)


if __name__ == "__main__":
    main()
