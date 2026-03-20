import torch


class RegionDecision:

    def __init__(self,region_weights):

        self.region_weights = region_weights

    def compute(self,risk):

        h = risk.shape[-2]

        num_regions = len(self.region_weights)

        step = int(h / num_regions)

        regions = []

        for i in range(num_regions):

            if i == num_regions - 1:
                r = risk[...,i*step:h,:]
            else:
                r = risk[...,i*step:(i+1)*step,:]

            regions.append(r)

        urgency = []

        for i,r in enumerate(regions):

            u = r.mean() * self.region_weights[i]

            urgency.append(u)

        urgency = torch.stack(urgency)

        decision = torch.argmax(urgency)

        return decision,urgency