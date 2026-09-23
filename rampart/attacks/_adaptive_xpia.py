# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Adaptive multi-turn XPIA execution."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from rampart.attacks._xpia import XPIAExecution
from rampart.core import (
    AgentAdapter,
    EvalOutcome,
    EvalResult,
    Payload,
    PromptDriver,
    Surface,
    Turn,
)
from rampart.core.execution import evaluate_turn_async

if TYPE_CHECKING:
    from rampart.core.evaluator import Evaluator
    from rampart.core.execution import ExecutionEventHandler

PayloadRewriter = Callable[[Payload, int, EvalResult], Payload | Awaitable[Payload]]


class AdaptiveXPIAExecution(XPIAExecution):
    """Re-inject adapted payloads while keeping the trigger benign."""

    def __init__(
        self,
        *,
        surface: Surface,
        payload: Payload,
        rewriter: PayloadRewriter,
        driver: PromptDriver,
        evaluator: Evaluator,
        max_attempts: int = 5,
        event_handlers: list[ExecutionEventHandler] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        super().__init__(
            handles=[],
            driver=driver,
            evaluator=evaluator,
            max_turns=max_attempts,
            event_handlers=event_handlers,
        )
        self._surface = surface
        self._initial_payload = payload
        self._rewriter = rewriter
        self._max_attempts = max_attempts

    @property
    def strategy_name(self) -> str:
        """Identifies adaptive XPIA executions in results."""
        return "xpia_adaptive"

    async def _run_phases_async(self, *, adapter: AgentAdapter) -> list[Turn]:
        turns: list[Turn] = []
        payload = self._initial_payload

        async with AsyncExitStack() as stack:
            session = await stack.enter_async_context(
                await adapter.create_session_async(),
            )

            for attempt in range(self._max_attempts):
                if attempt:
                    previous = turns[-1].eval_result
                    if previous is None or previous.outcome is not EvalOutcome.NOT_DETECTED:
                        break
                    payload = await self._rewrite_payload(
                        payload=payload,
                        attempt=attempt,
                        previous=previous,
                    )

                handle = self._surface.inject(payload=payload)
                self._handles.append(handle)
                async with handle:
                    await handle.wait_until_ready_async()
                    decision = await self._driver.next_prompt_async(history=turns)
                    if decision is None:
                        break

                    response = await session.send_async(decision.request)
                    turn = await evaluate_turn_async(
                        evaluator=self._evaluator,
                        history=turns,
                        request=decision.request,
                        response=response,
                        turn_number=len(turns),
                        driver_reasoning=decision.reasoning,
                        manifest=adapter.manifest,
                        observability_level=adapter.observability_profile,
                    )
                    turns.append(turn)
                    if turn.eval_result and turn.eval_result.detected:
                        break

        return turns

    async def _rewrite_payload(
        self,
        *,
        payload: Payload,
        attempt: int,
        previous: EvalResult,
    ) -> Payload:
        rewritten = self._rewriter(payload, attempt, previous)
        if inspect.isawaitable(rewritten):
            rewritten = await rewritten
        if not isinstance(rewritten, Payload):
            raise TypeError("payload rewriter must return a Payload")
        return rewritten
