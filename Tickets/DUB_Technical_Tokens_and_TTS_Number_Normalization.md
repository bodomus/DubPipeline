# DUB — Protect technical numeric tokens during translation and normalize numeric expressions before TTS

## Type

Bugfix / Reliability / Translation integrity / TTS preprocessing

## Priority

High

---

## Problem

The DubPipeline TTS stage crashes on some numeric expressions.

Observed failure:

```text
ValueError: invalid literal for int() with base 10: '1e-05'
```

Relevant stack fragment:

```text
num2words\__init__.py
num2words\lang_RU.py
return self._int2word(int(n), cardinal=True, case=case)

ValueError: invalid literal for int() with base 10: '1e-05'
```

The failing segment contains a decimal technical value:

```json
{
  "id": 320,
  "speaker": "SPEAKER_00",
  "start": 1778.415,
  "end": 1788.966,
  "text": "slope intensity this was just the default of one I'll go 0.0000001 and I'll actually change this to a",
  "text_tgt": "Сила наклона это был просто по умолчанию я пойду 0,00001 и я на самом деле изменю это на a",
  "text_ru": "Сила наклона это был просто по умолчанию я пойду 0,00001 и я на самом деле изменю это на a"
}
```

There are two separate defects:

1. The translation stage changes the numeric value:
   - source: `0.0000001`
   - translated result: `0,00001`

   These values are not equivalent.

2. The TTS preprocessing path eventually converts a decimal value into scientific notation such as:

   ```text
   1e-05
   ```

   and the Russian `num2words` backend fails when it tries to execute:

   ```python
   int("1e-05")
   ```

The fix must address both problems.

---

# Goal

Implement a reliable technical-token protection and TTS text normalization pipeline.

Target processing flow:

```text
source text
    ↓
protect technical tokens
    ↓
translation
    ↓
restore exact original technical tokens
    ↓
store translated text
    ↓
normalize text for TTS
    ↓
XTTS
```

The translation stage must not be able to silently alter protected numeric or technical values.

The TTS stage must not receive unsupported numeric forms that can trigger `num2words` failures.

---

# Mandatory constraints

## 1. Do not fix this by deleting numbers

Forbidden examples:

```python
re.sub(r"\d+", "", text)
```

or any equivalent behavior that removes technical values.

Technical values are part of the tutorial content and must remain semantically intact.

---

## 2. Do not fix this by replacing decimal separators only

Forbidden as a complete solution:

```python
text = text.replace(",", ".")
```

or:

```python
text = text.replace(".", ",")
```

Separator replacement does not solve:

- scientific notation;
- exact-value preservation across translation;
- negative values;
- percentages;
- dimensions;
- technical identifiers.

---

## 3. Do not modify the stored translated text only for TTS convenience

The pipeline must preserve a readable translated representation separately from the spoken TTS representation.

Preferred logical model:

```text
text
text_tgt
text_tts
```

`text_tts` may be transient and does not have to be persisted if the existing architecture does not require persistence.

The important requirement is separation of concerns:

```text
translated text != TTS-normalized text
```

Do not overwrite the translated text with spoken digit expansions unless the current storage architecture makes this unavoidable and the decision is justified.

---

# Scope A — Technical token protection around translation

## Requirement A1 — Protect numeric and technical tokens before translation

Introduce a dedicated component responsible for extracting and replacing technical tokens with placeholders before translation.

Example:

Input:

```text
I'll go 0.0000001 and then change it to -0.25
```

Protected:

```text
I'll go __TECH_TOKEN_0000__ and then change it to __TECH_TOKEN_0001__
```

Mapping:

```python
{
    "__TECH_TOKEN_0000__": "0.0000001",
    "__TECH_TOKEN_0001__": "-0.25",
}
```

After translation, restore exact original token values.

Expected result:

```text
я установлю 0.0000001, а затем изменю значение на -0.25
```

The exact token text must survive round-trip protection and restoration unchanged.

---

## Requirement A2 — Initial protected token classes

At minimum support:

### Decimal numbers

```text
0.0000001
0,00001
1.25
-0.25
+0.5
```

### Scientific notation

```text
1e-05
1E-5
2.5e+03
-3e-7
```

### Percentages

```text
50%
0.5%
-12.5%
```

### Dimensions / resolutions

```text
1920x1080
3840×2160
505x505
```

### Degree values

```text
45°
-15°
```

### Version-like numeric sequences

Examples:

```text
5.7
2.3
3.12
```

Be careful: version-like values and ordinary decimals may share syntax. Protection only needs to guarantee exact preservation; semantic classification is not required in the first implementation.

---

## Requirement A3 — Placeholder robustness

Placeholders must be designed so that translators are unlikely to:

- translate them;
- split them;
- alter case;
- insert whitespace;
- change numbering.

Use deterministic placeholders.

The implementation must detect restoration failures.

Do not silently continue if a protected token cannot be restored.

At minimum:

- log a warning or error with segment id/context;
- identify the missing or damaged placeholder;
- preserve diagnosability.

Prefer failing the affected translation segment explicitly rather than silently generating corrupted technical content.

---

## Requirement A4 — Restoration validation

After translation and restoration, validate that every protected token has been restored exactly once unless the implementation intentionally supports repeated placeholders.

Validation must detect:

- missing placeholder;
- duplicated placeholder;
- altered placeholder;
- unresolved placeholder remaining in translated output.

No unresolved token such as:

```text
__TECH_TOKEN_0000__
```

may reach TTS.

---

# Scope B — TTS text normalization

## Requirement B1 — Dedicated normalizer

Create a dedicated TTS text normalization component.

Suggested responsibility:

```python
normalize_text_for_tts(text: str, language: str) -> str
```

or an equivalent class-based design consistent with the existing project architecture.

Do not place a collection of ad hoc regex substitutions directly inside `step_tts.py`.

The normalizer must be independently testable.

---

## Requirement B2 — Russian decimal pronunciation

For Russian TTS, decimal technical values must be converted into a stable spoken form.

For technical tutorial content, prefer digit-by-digit pronunciation after the decimal separator.

Examples:

```text
0,00001
```

becomes:

```text
ноль запятая ноль ноль ноль ноль один
```

```text
0.0000001
```

becomes:

```text
ноль запятая ноль ноль ноль ноль ноль ноль один
```

```text
-0.25
```

becomes:

```text
минус ноль запятая два пять
```

The purpose is to make UI parameter values easy to reproduce.

Do not convert technical decimals into fraction names such as:

```text
одна стотысячная
```

for this first implementation.

---

## Requirement B3 — Scientific notation normalization

The normalizer must prevent raw scientific notation from reaching the failing `num2words` path.

Support at minimum:

```text
1e-05
1E-5
2.5e+03
-3e-7
```

Choose one deterministic spoken strategy and test it.

Recommended Russian pronunciation:

```text
1e-05
→ один умножить на десять в степени минус пять
```

```text
2.5e+03
→ два запятая пять умножить на десять в степени три
```

Alternative wording is acceptable only if:

- it is deterministic;
- it is understandable in technical tutorials;
- it avoids unsupported raw notation in XTTS;
- tests document the expected output.

---

## Requirement B4 — Preserve ordinary text

The normalizer must not damage ordinary text.

Examples that must remain structurally intact:

```text
Это обычное предложение.
Версия Unreal Engine 5.7 используется в проекте.
Размер текстуры 1920x1080.
Значение равно -0.25.
```

Only targeted technical tokens should be transformed.

---

## Requirement B5 — Language-aware behavior

Do not hardcode Russian normalization globally for all target languages.

The implementation must dispatch by target language or use a language-specific strategy.

At minimum:

```text
ru -> Russian normalizer
other languages -> unchanged text or existing behavior
```

Do not make Russian number words appear in non-Russian TTS.

---

# Scope C — Pipeline integration

## Requirement C1 — Translation integration

Integrate technical token protection around the actual translation operation.

Conceptually:

```python
protected_text, token_map = protector.protect(source_text)

translated = translator.translate(protected_text)

restored = protector.restore(
    translated,
    token_map,
)
```

Do not perform protection after translation. At that point the numeric value may already be corrupted.

---

## Requirement C2 — TTS integration point

Normalize the final target-language text immediately before passing it to XTTS.

Conceptually:

```python
tts_text = normalize_text_for_tts(
    translated_text,
    language=target_lang,
)

tts.tts_to_file(
    text=tts_text,
    ...
)
```

The normalizer must run before the Coqui XTTS call that currently reaches `num2words`.

---

## Requirement C3 — Logging

Add useful debug logging.

For a changed segment, log at debug level:

```text
TTS normalization segment_id=320
before="... 0,00001 ..."
after="... ноль запятая ноль ноль ноль ноль один ..."
```

Do not spam normal info-level output for every segment.

Warnings/errors are required for:

- failed placeholder restoration;
- unresolved placeholders;
- unsupported malformed numeric token if it would otherwise reach TTS.

Avoid logging huge full documents.

Segment-level context is sufficient.

---

# Tests

Tests are mandatory.

Do not consider the task complete with manual testing only.

---

## Test group 1 — Technical token protection

### Case 1

Input:

```text
value 0.0000001
```

After protect + simulated translation + restore:

```text
значение 0.0000001
```

Assert exact numeric preservation.

---

### Case 2

Input:

```text
set -0.25 and then 50%
```

Assert exact preservation of:

```text
-0.25
50%
```

---

### Case 3

Input:

```text
resolution 1920x1080
```

Assert exact resolution preservation.

---

### Case 4

Input:

```text
value 1e-05
```

Assert exact scientific notation preservation after restore.

---

### Case 5

Corrupt the placeholder deliberately in the simulated translated output.

Assert that restoration validation detects the corruption and does not silently return apparently valid text.

---

## Test group 2 — Russian TTS normalization

Required assertions:

```text
0,00001
→ ноль запятая ноль ноль ноль ноль один
```

```text
0.0000001
→ ноль запятая ноль ноль ноль ноль ноль ноль один
```

```text
-0.25
→ минус ноль запятая два пять
```

```text
1e-05
→ deterministic supported spoken form
```

```text
2.5e+03
→ deterministic supported spoken form
```

---

## Test group 3 — Regression for the observed crash

Add a regression test using the real failing text shape:

```text
Сила наклона это был просто по умолчанию я пойду 0,00001 и я на самом деле изменю это на a
```

Assert:

1. normalization completes without exception;
2. output does not contain raw `0,00001`;
3. output does not contain `1e-05`;
4. output contains a stable spoken Russian representation;
5. the original stored translated text is not accidentally mutated if the function contract is pure.

---

## Test group 4 — Translation-value integrity regression

Use source value:

```text
0.0000001
```

Simulate a translator that would otherwise change or localize surrounding text.

The protected/restored result must still contain exactly:

```text
0.0000001
```

and must never become:

```text
0,00001
```

This specific regression is mandatory because the observed pipeline already changed the value by two orders of magnitude.

---

# Acceptance Criteria

The task is complete only when all of the following are true:

- [ ] `0.0000001` survives translation with exactly the same numeric value.
- [ ] `0,00001` no longer causes the TTS stage to crash.
- [ ] `1e-05` no longer reaches the failing `num2words` path as raw unsupported input.
- [ ] Negative decimals are supported.
- [ ] Scientific notation is supported.
- [ ] Percentages are protected during translation.
- [ ] Resolutions such as `1920x1080` are protected during translation.
- [ ] No unresolved placeholders can silently reach TTS.
- [ ] Russian TTS normalization is language-specific.
- [ ] The normalizer is independently testable.
- [ ] Regression tests cover the exact observed failure.
- [ ] Existing TTS tests pass.
- [ ] Existing translation tests pass.
- [ ] Full project test suite passes.

---

# Non-goals

Do not expand this ticket into a complete linguistic number normalization system.

Not required in this task:

- perfect grammatical inflection for every number;
- currency normalization;
- dates;
- phone numbers;
- mathematical formula parsing;
- arbitrary unit conversion;
- locale-aware pronunciation for every supported language;
- rewriting translation architecture unrelated to token protection.

Keep the implementation focused and extensible.

---

# Implementation guidance

Prefer small focused components, for example:

```text
dubpipeline/
    text/
        technical_tokens.py
        tts_normalizer.py
```

or the closest structure consistent with the current repository.

Possible interfaces:

```python
class TechnicalTokenProtector:
    def protect(self, text: str) -> ProtectedText:
        ...

    def restore(self, text: str, tokens: TokenMap) -> str:
        ...
```

```python
def normalize_text_for_tts(
    text: str,
    language: str,
) -> str:
    ...
```

This is guidance, not a mandatory filename or exact class design.

Reuse existing project conventions where appropriate.

---

# Required Codex workflow

Before editing:

1. Inspect the current translation flow.
2. Inspect the TTS call path from `step_tts.py` into `tts_core.py`.
3. Find existing text preprocessing and number normalization logic.
4. Find current tests related to translation and TTS.
5. Do not duplicate an existing abstraction if one already exists.

During implementation:

1. Make the smallest coherent architectural change.
2. Keep token protection separate from TTS pronunciation normalization.
3. Add focused unit tests.
4. Add the exact crash regression test.
5. Run targeted tests first.
6. Run the complete project test suite.

Before finishing, report:

```text
Summary
Files changed
Tests added
Commands executed
Test results
Known limitations
```

Do not report the task as complete while any required regression test is missing.

---

# Expected result

The original failing segment must proceed through the pipeline without TTS failure.

The value:

```text
0.0000001
```

must remain numerically intact through translation.

Before Russian XTTS synthesis, a decimal technical value must be converted to a pronounceable representation such as:

```text
ноль запятая ноль ноль ноль ноль ноль ноль один
```

and raw scientific notation such as:

```text
1e-05
```

must not cause the pipeline to crash.
