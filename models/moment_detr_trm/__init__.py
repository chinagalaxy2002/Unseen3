"""
Moment-DETR-TRM Package
"""
from models.moment_detr_trm.moment_detr_trm import MomentDETR_TRM, build_moment_detr_trm
from models.moment_detr_trm.trm_modules import AttentivePooling, PhraseSlotMatcher
from models.moment_detr_trm.trm_loss import SetCriterionTRM, build_criterion_trm

__all__ = [
    "MomentDETR_TRM",
    "build_moment_detr_trm",
    "AttentivePooling",
    "PhraseSlotMatcher",
    "SetCriterionTRM",
    "build_criterion_trm",
]
