import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class CLIPMultiTaskModel(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", pretrained="openai"):
        super().__init__()

        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.visual_encoder = clip_model.visual
        feat_dim = getattr(self.visual_encoder, "output_dim", 512)

        # Disease head: 5 classes (healthy + 4 diseases)
        self.disease_pre_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.disease_classifier = nn.Linear(feat_dim, 5)
        self.logit_scale = nn.Parameter(torch.tensor(2.71).log())

        # Severity head
        self.severity_pre_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Fusion only: severity feature + disease probabilities + gating
        self.severity_head = nn.Sequential(
            nn.Linear(256 + 5 + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.severity_deltas = nn.Parameter(torch.zeros(4))
        self.severity_offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        feat = self.visual_encoder(x)

        d_feat = self.disease_pre_head(feat)
        d_feat = F.normalize(d_feat, dim=-1)

        disease_logits = self.disease_classifier(d_feat) * self.logit_scale.exp()
        disease_probs = torch.sigmoid(disease_logits)

        # Tính toán gating_signal từ lớp healthy (index 0)
        p_healthy = disease_probs[:, 0:1]
        gating_signal = 1.0 - p_healthy

        sev_feat = self.severity_pre_head(feat)

        # Kết hợp các đặc trưng cho severity
        sev_input = torch.cat([sev_feat, disease_probs, gating_signal], dim=-1)

        thresholds = torch.cumsum(
            F.softplus(self.severity_deltas), dim=0
        ) - self.severity_offset

        severity_score = self.severity_head(sev_input)
        severity_logits = severity_score - thresholds

        return {
            "disease_logits": disease_logits,
            "severity_logits": severity_logits,
            "disease_probs": disease_probs
        }
