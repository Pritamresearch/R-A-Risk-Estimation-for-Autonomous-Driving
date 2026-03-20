import torch.nn.functional as F


class FusionModule:

    def resize_depth(self, depth, seg):

        depth = F.interpolate(
            depth.unsqueeze(1),
            size=seg.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        assert depth.shape[-2:] == seg.shape[-2:]

        return depth