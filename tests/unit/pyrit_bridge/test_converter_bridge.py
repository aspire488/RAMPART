# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the generic PyRIT converter bridge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rampart.core.types import Payload, PayloadFormat
from rampart.pyrit_bridge.converter_bridge import PyRITConverterBridge, adapt_converter


def _text_payload(
    content: str = "test content",
    payload_id: str = "p-1",
    metadata: dict[str, str] | None = None,
) -> Payload:
    return Payload(content=content, id=payload_id, metadata=metadata or {})


def _result(output_text: str = "converted", output_type: str = "text") -> MagicMock:
    result = MagicMock()
    result.output_text = output_text
    result.output_type = output_type
    return result


class TestPyRITConverterBridge:
    """Generic text converter adaptation."""

    async def test_delegates_text_conversion(self) -> None:
        converter = MagicMock()
        converter.convert_async = AsyncMock(return_value=_result("encoded"))

        result = await PyRITConverterBridge(converter).convert_async(
            payload=_text_payload("hello"),
        )

        converter.convert_async.assert_awaited_once_with(
            prompt="hello",
            input_type="text",
        )
        assert result.content == "encoded"
        assert result.format is PayloadFormat.TEXT

    async def test_preserves_id_and_source_metadata(self) -> None:
        converter = MagicMock()
        converter.convert_async = AsyncMock(return_value=_result())
        source = _text_payload(payload_id="keep-me", metadata={"origin": "test"})

        result = await PyRITConverterBridge(converter).convert_async(payload=source)

        assert result.id == "keep-me"
        assert result.metadata["origin"] == "test"
        assert result.metadata["converter"] == "MagicMock"

    async def test_rejects_binary_input(self, tmp_path: Path) -> None:
        artifact = tmp_path / "input.docx"
        artifact.write_bytes(b"PK")
        payload = Payload(
            content="binary",
            format=PayloadFormat.DOCX,
            artifact=artifact,
        )
        converter = MagicMock()
        converter.convert_async = AsyncMock()

        with pytest.raises(ValueError, match="text payload"):
            await PyRITConverterBridge(converter).convert_async(payload=payload)

        converter.convert_async.assert_not_awaited()

    async def test_rejects_non_text_output(self) -> None:
        converter = MagicMock()
        converter.convert_async = AsyncMock(
            return_value=_result(output_type="image_path"),
        )

        with pytest.raises(ValueError, match="text output only"):
            await PyRITConverterBridge(converter).convert_async(
                payload=_text_payload(),
            )

    def test_rejects_object_without_convert_async(self) -> None:
        with pytest.raises(TypeError, match="convert_async"):
            PyRITConverterBridge(object())

    def test_adapt_converter_alias(self) -> None:
        converter = MagicMock()
        assert isinstance(adapt_converter(converter), PyRITConverterBridge)
