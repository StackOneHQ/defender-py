# Changelog

## [0.7.0](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.6.3...stackone-defender-v0.7.0) (2026-05-29)


### ⚠ BREAKING CHANGES

* The default ONNX model directory changed from minilm-full-aug to minilm-multihead-v5. Custom code that hardcoded the old path will no longer load.

### Features

* parity with TS defender 0.7.0 ([75d046a](https://github.com/StackOneHQ/stackone-defender/commit/75d046ab45066ee1f973e91357f7ecb23dea50c8))

## [0.6.3](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.6.2...stackone-defender-v0.6.3) (2026-05-26)


### ⚠ BREAKING CHANGES

* When `tier2_fields` is unset, Tier 2 scans all strings (no fallback to Tier 1 risky_field_names).

### Features

* align Python package with @stackone/defender 0.6.3 ([a91a904](https://github.com/StackOneHQ/stackone-defender/commit/a91a904de2a08a29479afb2cff31e8488468ebaf))


### Bug Fixes

* **ENG-269:** Python parity with @stackone/defender 0.6.3 ([7c312f1](https://github.com/StackOneHQ/stackone-defender/commit/7c312f1d1c858b2f25b49043d783ce7294638b82))


### Miscellaneous Chores

* prepare release 0.6.3 ([8ef9888](https://github.com/StackOneHQ/stackone-defender/commit/8ef9888752713ed5df76c4eed3e117605a8fb9e6))
* retrigger release workflow after gh actions outage ([72f586b](https://github.com/StackOneHQ/stackone-defender/commit/72f586bcb974b1aab08e7525253d9d8a9c8bc59d))

## [0.6.2](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.6.1...stackone-defender-v0.6.2) (2026-04-22)


### ⚠ BREAKING CHANGES

* Drop ToolSanitizationRule, config/sanitizer tool_rules, use_default_tool_rules, and get_tool_rule/should_skip_field. Matches @stackone/defender post ENG-12594.

### Features

* add missing functions for full TS API parity ([aec0c5b](https://github.com/StackOneHQ/stackone-defender/commit/aec0c5b8d31715df7e4ec2e4d306b55d595bb1c3))
* add PyPI publishing setup with Release Please CI ([2e28373](https://github.com/StackOneHQ/stackone-defender/commit/2e28373a27315dbb5e7deb23621977fe7fa2f7bc))
* add tier2_fields filter and export ToolSanitizationRule ([cb7fd93](https://github.com/StackOneHQ/stackone-defender/commit/cb7fd93fb88a30f40edc171ef3fcdc5d6ce2534d))
* align Python defender with Node (Tier 2 scoping, ONNX cache) ([482bfdd](https://github.com/StackOneHQ/stackone-defender/commit/482bfdda59b4617a75bc261621984cc321d28989))
* **ENG-12402:** add PyPI publishing setup with Release Please CI ([f979748](https://github.com/StackOneHQ/stackone-defender/commit/f979748a8a3b2084ea241c352866adcfcd0145ea))
* **ENG-12699:** TypeScript parity and synced ONNX bundle ([0449800](https://github.com/StackOneHQ/stackone-defender/commit/0449800fc2375c89ef231f5671f9a74bd84d3388))
* port stackone-defender from TypeScript to Python ([e3ff70d](https://github.com/StackOneHQ/stackone-defender/commit/e3ff70dd6a0bc94578dc4dbfde87c5d75f00b7b8))
* remove tool rules; batch Tier2 ONNX; lock ONNX load ([26c95c2](https://github.com/StackOneHQ/stackone-defender/commit/26c95c257175c892ae4be82ab7c17a099c1b6c6e))
* **sanitizer:** remove dead use_tier2_classification from ToolResultSanitizer ([4646179](https://github.com/StackOneHQ/stackone-defender/commit/46461798fcf5acc6ac6e23bc65177c35d9353d9c))
* sync Python package with TypeScript parity ([e1836dd](https://github.com/StackOneHQ/stackone-defender/commit/e1836dd967ad23997983ef1607118d1a25807e1c))
* upgrade ML classifier to jbv2 model (AgentShield 73.7 → 79.8) ([bcd27f8](https://github.com/StackOneHQ/stackone-defender/commit/bcd27f8abf954700276249f9b03de34f733c67c4))
* upgrade ML classifier to jbv5 (AgentShield 79.8 → 81.1) ([781dd10](https://github.com/StackOneHQ/stackone-defender/commit/781dd1007e7a0db03d58619a23b69f1b5d73e85d))


### Bug Fixes

* address Copilot/cubic review (Tier2 scope, tokens, SFE, thresholds) ([bf173ac](https://github.com/StackOneHQ/stackone-defender/commit/bf173ac42f6aaa7513ea2a1fc19083806a5c5ee1))
* **ci:** avoid fasttext-wheel on Python 3.13 ([a6cda76](https://github.com/StackOneHQ/stackone-defender/commit/a6cda76894e3cd240c4f104e701e3202babb2682))
* **classifier:** surface classification errors in classify_by_sentence skip_reason ([bd94639](https://github.com/StackOneHQ/stackone-defender/commit/bd9463978dac5572f999d8ec3ed1adbaf0bb97f2))
* default enable_tier2 to True to match TypeScript SDK behaviour ([d66773b](https://github.com/StackOneHQ/stackone-defender/commit/d66773bee026517d09dd56b9311dd3c281c6f675))
* **defender:** fix _extract_strings filtering, None checks, and cache ONNX load failure ([bf4ce99](https://github.com/StackOneHQ/stackone-defender/commit/bf4ce993287db9e067b661100b5bd92cc21aef6b))
* **defender:** sync hasThreats blocking logic and tool rules precedence from JS package ([a217c3e](https://github.com/StackOneHQ/stackone-defender/commit/a217c3ef27aa0e4d92f21571bf0559ff9906f660))
* enable tier2 by default to match TypeScript package ([f1fe990](https://github.com/StackOneHQ/stackone-defender/commit/f1fe990e1a81c32cb271f6ca85cc063f3da49223))
* sync Python with TypeScript parity ([cec0813](https://github.com/StackOneHQ/stackone-defender/commit/cec0813ff8cc98f4502d5916d285a28877983d98))
* **tier2:** apply max_text_length truncation in classify_by_sentence ([a67d2c6](https://github.com/StackOneHQ/stackone-defender/commit/a67d2c6524fb1d6b4f9331f547f28221867038de))
* upgrade ML classifier to jbv2 (AgentShield 73.7 → 79.8) ([b452b39](https://github.com/StackOneHQ/stackone-defender/commit/b452b39c718329355f50c418bd50c37da2ed8698))
* upgrade ML classifier to jbv2 (AgentShield 73.7 → 79.8) ([ccb1204](https://github.com/StackOneHQ/stackone-defender/commit/ccb1204d5e3d9763bb916d71bb49b75039ceb197))
* use uv instead of pip in README installation instructions ([519759f](https://github.com/StackOneHQ/stackone-defender/commit/519759f09c6fc1eb6bf97f53ad0cbd25c78e2893))


### Dependencies

* **sfe:** switch optional FastText bindings to fasttext-ng ([bc9cc28](https://github.com/StackOneHQ/stackone-defender/commit/bc9cc283bc2da9f10472415d4aa94a0df083ec3d))


### Documentation

* add README adapted from TypeScript package ([a03c757](https://github.com/StackOneHQ/stackone-defender/commit/a03c757a1760b797d9a3ef444950e2839ca1c52d))
* update README — enable_tier2 defaults to True ([af0d059](https://github.com/StackOneHQ/stackone-defender/commit/af0d05957e39a83b7e6e18b1f78b95219b14a4f5))
* update README to reflect changes in package name and Python version ([d2fc2ca](https://github.com/StackOneHQ/stackone-defender/commit/d2fc2ca1900e2f6410df2ec075c5a8a1c3ac241b))


### Miscellaneous Chores

* prepare patch release 0.6.2 ([7b3c105](https://github.com/StackOneHQ/stackone-defender/commit/7b3c105b2ce23f88f284d72e41c1917aefdc4537))

## [0.6.1](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.1.2...stackone-defender-v0.6.1) (2026-04-21)

### Features

* align Python package behavior with `@stackone/defender` 0.6.1
* add SFE preprocessing support (`use_sfe`) with fail-open optional runtime loading
* add packed-chunk Tier 2 batching and density-adjusted scoring
* add dangerous-key traversal hardening (`__proto__`, `constructor`, `prototype`)
* add cumulative-risk fractional thresholds to reduce list-response false positives

### Bug Fixes

* use `fasttext-ng` instead of `fasttext-wheel` for the `[sfe]` extra and dev tests so Python 3.13 CI can install maintained FastText bindings (NumPy 2.3+).

### Breaking Changes

* Python package version jumps from `0.1.2` to `0.6.1` to align release train with TypeScript parity.
* `DefenseResult` now includes `fields_dropped` and `truncated_at_depth`.

## [0.1.2](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.1.1...stackone-defender-v0.1.2) (2026-04-08)


### Bug Fixes

* upgrade ML classifier to jbv2 (AgentShield 73.7 → 79.8) ([b452b39](https://github.com/StackOneHQ/stackone-defender/commit/b452b39c718329355f50c418bd50c37da2ed8698))


### Documentation

* update README to reflect changes in package name and Python version ([d2fc2ca](https://github.com/StackOneHQ/stackone-defender/commit/d2fc2ca1900e2f6410df2ec075c5a8a1c3ac241b))

## [0.1.1](https://github.com/StackOneHQ/stackone-defender/compare/stackone-defender-v0.1.0...stackone-defender-v0.1.1) (2026-04-08)


### Features

* add missing functions for full TS API parity ([aec0c5b](https://github.com/StackOneHQ/stackone-defender/commit/aec0c5b8d31715df7e4ec2e4d306b55d595bb1c3))
* add PyPI publishing setup with Release Please CI ([2e28373](https://github.com/StackOneHQ/stackone-defender/commit/2e28373a27315dbb5e7deb23621977fe7fa2f7bc))
* add tier2_fields filter and export ToolSanitizationRule ([cb7fd93](https://github.com/StackOneHQ/stackone-defender/commit/cb7fd93fb88a30f40edc171ef3fcdc5d6ce2534d))
* **ENG-12402:** add PyPI publishing setup with Release Please CI ([f979748](https://github.com/StackOneHQ/stackone-defender/commit/f979748a8a3b2084ea241c352866adcfcd0145ea))
* port stackone-defender from TypeScript to Python ([e3ff70d](https://github.com/StackOneHQ/stackone-defender/commit/e3ff70dd6a0bc94578dc4dbfde87c5d75f00b7b8))
* **sanitizer:** remove dead use_tier2_classification from ToolResultSanitizer ([4646179](https://github.com/StackOneHQ/stackone-defender/commit/46461798fcf5acc6ac6e23bc65177c35d9353d9c))
* sync Python package with TypeScript parity ([e1836dd](https://github.com/StackOneHQ/stackone-defender/commit/e1836dd967ad23997983ef1607118d1a25807e1c))


### Bug Fixes

* **classifier:** surface classification errors in classify_by_sentence skip_reason ([bd94639](https://github.com/StackOneHQ/stackone-defender/commit/bd9463978dac5572f999d8ec3ed1adbaf0bb97f2))
* **defender:** fix _extract_strings filtering, None checks, and cache ONNX load failure ([bf4ce99](https://github.com/StackOneHQ/stackone-defender/commit/bf4ce993287db9e067b661100b5bd92cc21aef6b))
* **defender:** sync hasThreats blocking logic and tool rules precedence from JS package ([a217c3e](https://github.com/StackOneHQ/stackone-defender/commit/a217c3ef27aa0e4d92f21571bf0559ff9906f660))
* enable tier2 by default to match TypeScript package ([f1fe990](https://github.com/StackOneHQ/stackone-defender/commit/f1fe990e1a81c32cb271f6ca85cc063f3da49223))
* sync Python with TypeScript parity ([cec0813](https://github.com/StackOneHQ/stackone-defender/commit/cec0813ff8cc98f4502d5916d285a28877983d98))
* use uv instead of pip in README installation instructions ([519759f](https://github.com/StackOneHQ/stackone-defender/commit/519759f09c6fc1eb6bf97f53ad0cbd25c78e2893))


### Documentation

* add README adapted from TypeScript package ([a03c757](https://github.com/StackOneHQ/stackone-defender/commit/a03c757a1760b797d9a3ef444950e2839ca1c52d))

## Changelog
