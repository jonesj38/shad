import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from shad.engine.llm import LLMProvider, ModelTier
from shad.utils.config import Settings

@pytest.mark.asyncio
async def test_gemini_cli_timeout():
    """Test that Gemini CLI calls timeout correctly."""
    
    # Mock settings with a short timeout
    mock_settings = MagicMock(spec=Settings)
    mock_settings.gemini_cli_timeout = 1
    mock_settings.gemini_orchestrator_model = "gemini-3-pro-preview"
    mock_settings.gemini_worker_model = "gemini-3-pro-preview"
    mock_settings.gemini_leaf_model = "gemini-3-flash-preview"
    mock_settings.use_gemini_cli = True
    mock_settings.anthropic_api_key = ""
    mock_settings.openai_api_key = ""

    with patch("shad.engine.llm.get_settings", return_value=mock_settings):
        provider = LLMProvider(use_gemini_cli=True)

        # Create a mock process that hangs
        mock_process = AsyncMock()
        mock_process.kill = MagicMock()
        mock_process.returncode = 0
        
        # Make communicate sleep longer than the timeout
        async def delayed_communicate(*args, **kwargs):
            await asyncio.sleep(2)
            return (b"stdout", b"stderr")
            
        mock_process.communicate.side_effect = delayed_communicate
        
        # Mock create_subprocess_exec
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            with pytest.raises(RuntimeError) as exc_info:
                await provider._complete_gemini_cli("test prompt")
            
            # Verify the error message
            assert "Gemini CLI timed out" in str(exc_info.value)
            
            # Verify process was killed
            mock_process.kill.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_cli_success():
    """Test successful Gemini CLI call."""
    
    mock_settings = MagicMock(spec=Settings)
    mock_settings.gemini_cli_timeout = 5
    mock_settings.use_gemini_cli = True
    
    with patch("shad.engine.llm.get_settings", return_value=mock_settings):
        provider = LLMProvider(use_gemini_cli=True)
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"response text", b"")
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            response, _ = await provider._complete_gemini_cli("test prompt")
            assert response == "response text"
