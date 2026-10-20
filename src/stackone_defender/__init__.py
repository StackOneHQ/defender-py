"""stackone-defender: Prompt injection defense for AI tool-calling.

Usage:
    from stackone_defender import create_prompt_defense

    defense = create_prompt_defense(enable_tier2=True)
    defense.warmup_tier2()

    result = defense.defend_tool_result(tool_output, "gmail_get_message")
    if not result.allowed:
        print(f"Blocked: {result.risk_level}")
"""

from .classifiers.onnx_classifier import get_default_model_path
from .classifiers.tier3_llm_provider import (
    DEFAULT_TIER3_PROMPT,
    TIER3_PROMPT_VERSION,
    VERDICT_JSON_SCHEMA,
    LlmCompletionRequest,
    PromptInput,
    Tier3PromptTemplate,
    build_tier3_messages,
    create_llm_tier3_provider,
    parse_verdict_json,
)
from .classifiers.tier3_openai_compatible import create_openai_compatible_tier3_provider
from .classifiers.tier3_orchestrator import get_default_tier3_provider, set_default_tier3_provider
from .core.prompt_defense import PromptDefense, create_prompt_defense
from .sfe.preprocess import (
    DropDecision,
    SfePredictor,
    SfePreprocessResult,
    get_default_predictor,
    get_default_sfe_model_path,
    sfe_preprocess,
)
from .types import (
    DefenderMode,
    DefenseResult,
    MultiheadConfig,
    RiskLevel,
    Tier1Result,
    Tier3ChunkSummary,
    Tier3Provider,
    Tier3TokenUsage,
    Tier3Verdict,
)
from .utils.boundary import contains_boundary_patterns, generate_boundary_instructions

__all__ = [
    "DEFAULT_TIER3_PROMPT",
    "TIER3_PROMPT_VERSION",
    "VERDICT_JSON_SCHEMA",
    "DefenderMode",
    "DefenseResult",
    "DropDecision",
    "LlmCompletionRequest",
    "MultiheadConfig",
    "PromptDefense",
    "PromptInput",
    "RiskLevel",
    "SfePredictor",
    "SfePreprocessResult",
    "Tier1Result",
    "Tier3ChunkSummary",
    "Tier3PromptTemplate",
    "Tier3Provider",
    "Tier3TokenUsage",
    "Tier3Verdict",
    "build_tier3_messages",
    "contains_boundary_patterns",
    "create_llm_tier3_provider",
    "create_openai_compatible_tier3_provider",
    "create_prompt_defense",
    "generate_boundary_instructions",
    "get_default_model_path",
    "get_default_predictor",
    "get_default_sfe_model_path",
    "get_default_tier3_provider",
    "parse_verdict_json",
    "set_default_tier3_provider",
    "sfe_preprocess",
]
