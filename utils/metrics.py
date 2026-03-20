import torch

def collision_risk_score(risk_map,threshold=0.6):

    danger = (risk_map > threshold).float()

    return danger.mean()