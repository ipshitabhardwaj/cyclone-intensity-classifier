"""
Minimal Grad-CAM implementation, hook-based, targeting `model.target_layer`
(the last conv block of the EfficientNet backbone). Produces a heatmap over
the input showing which spatial regions drove the predicted intensity
category — this is the forecaster-facing "why did the model say this"
explanation.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.target_layer
        self._activations = None
        self._gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        x: (1, C, H, W) single input tensor
        Returns: (heatmap [H,W] in [0,1], predicted_class_idx, predicted_probs)
        """
        self.model.eval()
        cls_out, reg_out = self.model(x)
        probs = F.softmax(cls_out, dim=1)
        if class_idx is None:
            class_idx = int(cls_out.argmax(dim=1).item())

        self.model.zero_grad()
        score = cls_out[0, class_idx]
        score.backward(retain_graph=True)

        grads = self._gradients[0]        # (C, h, w)
        acts = self._activations[0]       # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # global-average-pool the gradients -> channel weights

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * acts[c]
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        # upsample to input resolution
        cam_t = torch.from_numpy(cam)[None, None]
        cam_up = F.interpolate(cam_t, size=x.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = cam_up[0, 0].numpy()
        return heatmap, class_idx, probs.detach().cpu().numpy()[0], reg_out.detach().cpu().numpy()[0]


def overlay_heatmap(base_gray: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """base_gray in [0,1], heatmap in [0,1] -> RGB uint8 image with heatmap overlay."""
    import matplotlib.cm as cm
    colored = cm.get_cmap("jet")(heatmap)[..., :3]  # (H,W,3) in [0,1]
    base_rgb = np.stack([base_gray] * 3, axis=-1)
    blended = 0.55 * base_rgb + 0.45 * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
