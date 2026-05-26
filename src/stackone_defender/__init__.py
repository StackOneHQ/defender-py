"""stackone-defender: Prompt injection defense for AI tool-calling.

Usage:
    from stackone_defender import create_prompt_defense

    defense = create_prompt_defense(enable_tier2=True)
    defense.warmup_tier2()

    result = defense.defend_tool_result(tool_output, "gmail_get_message")
    if not result.allowed:
        print(f"Blocked: {result.risk_level}")
"""

from .core.prompt_defense import PromptDefense, create_prompt_defense
from .utils.boundary import contains_boundary_patterns, generate_boundary_instructions
from .sfe.preprocess import (
    DropDecision,
    SfePredictor,
    SfePreprocessResult,
    get_default_predictor,
    get_default_sfe_model_path,
    sfe_preprocess,
)
from .types import DefenseResult, RiskLevel, Tier1Result

__all__ = [
    "DefenseResult",
    "DropDecision",
    "PromptDefense",
    "RiskLevel",
    "SfePredictor",
    "SfePreprocessResult",
    "Tier1Result",
    "contains_boundary_patterns",
    "create_prompt_defense",
    "generate_boundary_instructions",
    "get_default_predictor",
    "get_default_sfe_model_path",
    "sfe_preprocess",
]
