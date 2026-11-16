from __future__ import annotations
import torch
import torch.nn.functional as F
from clingeo.model import FoundationBiometry
from clingeo.models.clingeo_usbiometry import decode_heatmap_with_offset


class PreciseFoundationBiometry(FoundationBiometry):
    """Keep geometric heads and coordinate expectations in float32 under AMP."""

    def forward(self, image, task_id):
        image = (image[:, :1].expand(-1, 3, -1, -1) - self.input_mean) / self.input_std
        embedding = self.task_embedding(task_id)
        dense = self.detail(self.decoder(self.backbone(image), embedding))
        with torch.autocast(device_type=image.device.type, enabled=False):
            dense, embedding = dense.float(), embedding.float()
            logits, offset = self.heatmap_head(dense), self.offset_head(dense)
            pooled = F.adaptive_avg_pool2d(dense, 1).flatten(1)
            query = torch.cat([pooled, embedding], dim=1)
            direct = self.coord_query_head(query).sigmoid().view(-1, self.max_landmarks, 2)
            direct = direct * float(self.input_size - 1)
            log_sigma = self.uncertainty_head(dense).view(-1, self.max_landmarks, 1).clamp(-5, 5)
            spatial, prob, peak, entropy = decode_heatmap_with_offset(
                logits, offset, self.heatmap_stride, self.input_size, self.max_landmarks,
            )
            task = embedding[:, None].expand(-1, self.max_landmarks, -1)
            gate = self.gate_head(torch.cat([
                task, peak[..., None], entropy[..., None], log_sigma.exp(),
            ], -1)).sigmoid()
            return {
                "heatmap": logits, "heatmap_prob": prob, "offset": offset,
                "coord_heatmap": spatial, "coord_query": direct,
                "coords": gate * spatial + (1 - gate) * direct,
                "log_sigma": log_sigma, "gate": gate,
                "params": self.param_head(query)[:, :self.max_params],
            }
