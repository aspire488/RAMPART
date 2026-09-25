# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Adapt PyRIT prompt converters to RAMPART payload converters."""

from __future__ import annotations

from typing import Any

from rampart.core.types import Payload, PayloadFormat


class PyRITConverterBridge:
    """Adapt a PyRIT Converter to RAMPART's PayloadConverter protocol.

    The first version supports text-to-text converters. PyRIT-specific types
    remain outside RAMPART's public method signatures.
    """

    def __init__(self, converter: Any) -> None:
        """Initialize the bridge with a PyRIT converter instance."""
        if not callable(getattr(converter, "convert_async", None)):
            msg = "converter must provide an async convert_async method."
            raise TypeError(msg)
        self._converter: Any = converter

    async def convert_async(self, *, payload: Payload) -> Payload:
        """Convert a RAMPART text payload with the wrapped PyRIT converter."""
        if not payload.format.is_text:
            msg = (
                "PyRITConverterBridge requires a text payload, "
                f"got {payload.format.value}."
            )
            raise ValueError(msg)

        result = await self._converter.convert_async(
            prompt=payload.content,
            input_type="text",
        )

        if result.output_type != "text":
            msg = (
                "PyRITConverterBridge currently supports text output only, "
                f"got {result.output_type!r}."
            )
            raise ValueError(msg)

        metadata = {
            **payload.metadata,
            "converter": self._converter.__class__.__name__,
        }
        return Payload(
            content=result.output_text,
            id=payload.id,
            format=PayloadFormat.TEXT,
            metadata=metadata,
        )


adapt_converter = PyRITConverterBridge

__all__ = ["PyRITConverterBridge", "adapt_converter"]
