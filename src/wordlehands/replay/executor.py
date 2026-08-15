"""Deterministic replay (Section 3.3): given a saved Capability artifact and
input params, drive the Surface using ONLY the artifact's recorded
LocatorSpecs — no LLM in the decision loop — assert the checkpoint, and
return a structured ReplayResult.

Error handling strategy: a step that fails is retried once (bounded — this is
the "transient slowness" recoverable case), then hard-fails with full
evidence. Once all steps have executed, the checkpoint is checked; if it does
NOT hold, the executor checks the capability's error_taxonomy for a known
explanation (e.g. "not a valid word") before concluding it's an
undiagnosed hard failure. This ordering — steps, then checkpoint, then
taxonomy-as-explanation-of-checkpoint-failure — is what keeps "no such
member"-style business outcomes from ever surfacing as crashes.
"""

from __future__ import annotations

import asyncio
import re
import time

from wordlehands.artifact.schema import (
    Capability,
    CheckpointCondition,
    ExtractField,
    OutcomeCategory,
    OutputType,
    ParamSpec,
    Step,
    StepAction,
)
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.replay.results import ReplayBusinessOutcome, ReplayFailure, ReplayResult, ReplaySuccess
from wordlehands.surface.base import Action, ActionType, Surface


class InputValidationError(Exception):
    pass


class ReplayExecutor:
    def __init__(self, surface: Surface, evidence: EvidenceLogger):
        self.surface = surface
        self.evidence = evidence

    async def run(self, capability: Capability, inputs: dict) -> ReplayResult:
        self._validate_inputs(capability.inputs, inputs)
        self.evidence.log(
            "replay_start",
            capability_id=capability.capability_id,
            version=capability.version,
            inputs=inputs,
        )

        # Precondition check: reuse the error taxonomy to catch states where
        # running the steps at all would be meaningless (e.g. the game is
        # already over from a prior call) — cheaper and safer than executing
        # steps against a surface we already know is in the wrong state.
        pre_entry, pre_detail = await self._match_taxonomy(capability)
        if pre_entry is not None:
            self.evidence.log(
                "precondition_short_circuit", code=pre_entry.code, detail=pre_detail
            )
            if pre_entry.category == OutcomeCategory.HARD_FAILURE:
                return await self._fail(capability, None, pre_entry.description, pre_detail, pre_detail)
            result = ReplayBusinessOutcome(
                capability_id=capability.capability_id,
                version=capability.version,
                code=pre_entry.code,
                detail=pre_detail,
            )
            self.evidence.write_result(result.model_dump())
            return result

        for step in capability.steps:
            outcome = await self._execute_step(step, inputs)
            self.evidence.log(
                "step_executed",
                step_id=step.step_id,
                action=step.action.value,
                ok=outcome.ok,
                message=outcome.message,
            )
            if outcome.blocked_by_guardrail:
                return await self._fail(
                    capability, step.step_id, "action permitted by guardrail policy",
                    outcome.message, outcome.message,
                )
            if not outcome.ok:
                # bounded single retry for transient conditions
                await asyncio.sleep(0.3)
                retry_outcome = await self._execute_step(step, inputs)
                self.evidence.log(
                    "step_retry",
                    step_id=step.step_id,
                    ok=retry_outcome.ok,
                    message=retry_outcome.message,
                )
                if not retry_outcome.ok:
                    return await self._fail(
                        capability, step.step_id, "step to execute successfully",
                        retry_outcome.message, retry_outcome.message,
                    )

        checkpoint_ok, checkpoint_detail = await self._poll_checkpoint(capability.checkpoint)
        self.evidence.log(
            "checkpoint_checked", ok=checkpoint_ok, detail=checkpoint_detail
        )

        if not checkpoint_ok:
            entry, detail = await self._match_taxonomy(capability)
            if entry is not None:
                self.evidence.log(
                    "error_taxonomy_matched", code=entry.code, category=entry.category.value, detail=detail
                )
                if entry.category == OutcomeCategory.HARD_FAILURE:
                    return await self._fail(capability, None, entry.description, detail, detail)
                result = ReplayBusinessOutcome(
                    capability_id=capability.capability_id,
                    version=capability.version,
                    code=entry.code,
                    detail=detail,
                )
                self.evidence.write_result(result.model_dump())
                return result

            return await self._fail(
                capability,
                None,
                capability.checkpoint.description,
                checkpoint_detail,
                "checkpoint not met and no known error_taxonomy entry explains why",
            )

        outputs = await self._extract_outputs(capability.outputs)
        result = ReplaySuccess(
            capability_id=capability.capability_id, version=capability.version, outputs=outputs
        )
        self.evidence.write_result(result.model_dump())
        return result

    def _validate_inputs(self, params: list[ParamSpec], inputs: dict) -> None:
        for p in params:
            if p.required and p.name not in inputs:
                raise InputValidationError(f"missing required input '{p.name}'")
            if p.name in inputs and p.pattern:
                value = str(inputs[p.name])
                if not re.fullmatch(p.pattern, value):
                    raise InputValidationError(
                        f"input '{p.name}'={value!r} does not match pattern {p.pattern!r}"
                    )

    async def _execute_step(self, step: Step, inputs: dict):
        if step.action == StepAction.PRESS_KEY:
            action = Action(type=ActionType.PRESS_KEY, key=step.literal_key, reason=step.description)
        elif step.action == StepAction.TYPE_TEXT:
            text = str(inputs[step.input_param]) if step.input_param else ""
            action = Action(type=ActionType.TYPE_TEXT, text=text, reason=step.description)
        elif step.action == StepAction.CLICK:
            action = Action(type=ActionType.CLICK, locator=step.locator, reason=step.description)
        elif step.action == StepAction.WAIT:
            action = Action(type=ActionType.WAIT, wait_ms=step.wait_ms, reason=step.description)
        else:
            raise ValueError(f"unsupported step action {step.action}")
        return await self.surface.act(action)

    async def _poll_checkpoint(self, checkpoint) -> tuple[bool, str]:
        deadline = time.monotonic() + checkpoint.timeout_ms / 1000
        last_detail = "never evaluated"
        while time.monotonic() < deadline:
            ok, detail = await self._check_checkpoint_once(checkpoint)
            last_detail = detail
            if ok:
                return True, detail
            await asyncio.sleep(0.15)
        return False, last_detail

    async def _check_checkpoint_once(self, checkpoint) -> tuple[bool, str]:
        resolved = await self.surface.resolve(checkpoint.locator, attribute=checkpoint.attribute, each=True)
        if not resolved.found:
            return False, "locator did not resolve"

        if checkpoint.condition == CheckpointCondition.ALL_MATCH_NONEMPTY_ATTR:
            ok = len(resolved.values) > 0 and all(v.strip() != "" for v in resolved.values)
            return ok, f"values={resolved.values}"

        if checkpoint.condition == CheckpointCondition.ALL_CONTAIN_SUBSTRING:
            match = checkpoint.match or ""
            ok = len(resolved.values) > 0 and all(match in v for v in resolved.values)
            return ok, f"values={resolved.values}"

        if checkpoint.condition == CheckpointCondition.TEXT_PRESENT:
            match = checkpoint.match or ""
            ok = any(match in v for v in resolved.values)
            return ok, f"values={resolved.values}"

        return False, f"unsupported condition {checkpoint.condition}"

    async def _match_taxonomy(self, capability: Capability):
        for entry in capability.error_taxonomy:
            matched, detail = await self._check_error_entry(entry)
            if matched:
                return entry, detail
        return None, ""

    async def _check_error_entry(self, entry) -> tuple[bool, str]:
        resolved = await self.surface.resolve(entry.locator, attribute="text")
        if not resolved.found:
            return False, ""
        for v in resolved.values:
            if entry.match_text in v:
                return True, v
        return False, resolved.values[0] if resolved.values else ""

    async def _extract_outputs(self, fields: list[ExtractField]) -> dict:
        outputs: dict = {}
        for field in fields:
            resolved = await self.surface.resolve(field.locator, attribute=field.attribute, each=field.each)
            values = resolved.values if resolved.found else []

            if field.type == OutputType.ENUM and field.enum_values:
                values = [self._map_enum(v, field.enum_values) for v in values]

            if field.each:
                outputs[field.name] = values
            else:
                outputs[field.name] = values[0] if values else None
        return outputs

    @staticmethod
    def _map_enum(raw: str, enum_values: list[str]) -> str:
        for ev in enum_values:
            if ev in raw:
                return ev
        return raw

    async def _fail(
        self, capability: Capability, step_id: str | None, expected: str, observed: str, message: str
    ) -> ReplayFailure:
        try:
            obs = await self.surface.observe()
            self.evidence.save_failure_bundle(obs, message)
        except Exception:
            pass
        result = ReplayFailure(
            capability_id=capability.capability_id,
            version=capability.version,
            step_id=step_id,
            expected=expected,
            observed=observed,
            message=message,
        )
        self.evidence.write_result(result.model_dump())
        return result
