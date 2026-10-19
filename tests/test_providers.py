import json
import importlib.resources
from pathlib import Path

import pytest

from chiplet_layout_workflow.prompts import PromptError, PromptRenderer
from chiplet_layout_workflow.providers import (
    ArtifactError,
    LLMResponse,
    MockProvider,
    OpenAIProvider,
    ProviderConfigError,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def test_mock_provider_returns_configured_deterministic_json():
    provider = MockProvider({"decision": "accept", "confidence": 0.9})

    result = provider.generate(
        system_prompt="system",
        user_prompt="user",
        artifacts=[],
    )

    assert isinstance(result, LLMResponse)
    assert result.text == '{"decision": "accept", "confidence": 0.9}'
    assert result.model == "mock"
    assert result.provider_request_id == "mock-request"


def test_mock_provider_never_uses_network(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    result = MockProvider({"ok": True}).generate("s", "u", [], None)

    assert json.loads(result.text) == {"ok": True}


def test_prompt_renderer_renders_role_prompt(tmp_path):
    (tmp_path / "reader.md").write_text("Read {topic} as evidence.", encoding="utf-8")

    rendered = PromptRenderer(tmp_path).render("reader", {"topic": "layout"})

    assert rendered == "Read layout as evidence."


def test_prompt_renderer_rejects_missing_role_prompt(tmp_path):
    with pytest.raises(PromptError, match="reader"):
        PromptRenderer(tmp_path).render("reader", {})


def test_prompt_renderer_rejects_role_path_traversal(tmp_path):
    secret = tmp_path / "secret.md"
    secret.write_text("do not read", encoding="utf-8")

    with pytest.raises(PromptError, match="Unknown role"):
        PromptRenderer(tmp_path).render("../secret", {})


@pytest.mark.parametrize("role", ["reader", "retriever", "builder", "verifier", "version_controller"])
def test_repository_role_prompts_define_safe_boundaries(role):
    prompt = PromptRenderer(PROMPTS_DIR).render(role, {})

    assert "Role boundary:" in prompt
    assert "Evidence versus inference:" in prompt
    assert "Required output files:" in prompt
    assert "untrusted data" in prompt


def test_repository_role_prompts_are_independent_of_working_directory(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent)

    prompt = PromptRenderer(PROMPTS_DIR).render("reader", {})

    assert "Role boundary:" in prompt


def test_default_prompt_renderer_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    prompt = PromptRenderer().render("reader", {})

    assert "Role boundary:" in prompt


@pytest.mark.parametrize("role", ["reader", "retriever", "builder", "verifier", "version_controller"])
def test_prompt_assets_are_available_as_package_resources(role):
    resource = importlib.resources.files("chiplet_layout_workflow.prompt_assets").joinpath(
        f"{role}.md"
    )

    assert resource.is_file()
    assert "Role boundary:" in resource.read_text(encoding="utf-8")


@pytest.mark.parametrize("role", ["reader", "retriever", "builder", "verifier", "version_controller"])
def test_packaged_prompt_assets_match_source_templates(role):
    source = (PROMPTS_DIR / f"{role}.md").read_text(encoding="utf-8")
    packaged = importlib.resources.files("chiplet_layout_workflow.prompt_assets").joinpath(
        f"{role}.md"
    ).read_text(encoding="utf-8")

    assert packaged == source


def test_reader_prompts_define_structured_envelope_and_target_boundary():
    prompt_paths = (
        PROMPTS_DIR / "reader.md",
        Path(
            importlib.resources.files("chiplet_layout_workflow.prompt_assets")
            .joinpath("reader.md")
        ),
    )
    required_fields = (
        "paper_understanding",
        "core_method",
        "candidate_plan",
        "transferable_ideas",
        "limitations",
        "citations",
        "target_assessment",
    )
    required_files = (
        "paper_understanding.md",
        "core_method.md",
        "candidate_plan.md",
        "transferable_ideas.md",
        "limitations.md",
        "citations.json",
        "target_assessment.json",
    )

    for prompt_path in prompt_paths:
        prompt = prompt_path.read_text(encoding="utf-8")
        assert "one structured JSON envelope" in prompt
        assert all(field in prompt for field in required_fields)
        assert all(filename in prompt for filename in required_files)
        assert "evidence" in prompt.lower() and "inference" in prompt.lower()
        assert all(status in prompt for status in ("confirmed", "uncertain", "needs_change"))
        assert all(
            marker in prompt
            for marker in ("pending", "unapproved", "requires_human_approval")
        )
        assert "does not apply target changes" in prompt
        assert "does not run complete experiments" in prompt


def test_openai_provider_rejects_missing_api_key_before_client_creation(monkeypatch):
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    created = False

    def fail_client(*args, **kwargs):
        nonlocal created
        created = True
        raise AssertionError("client must not be created")

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", fail_client)

    with pytest.raises(ProviderConfigError, match="TEST_OPENAI_KEY"):
        OpenAIProvider(api_key_env="TEST_OPENAI_KEY")

    assert created is False


def test_openai_provider_accepts_configured_model_timeout_and_env_name(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)

    OpenAIProvider(api_key_env_var="TEST_OPENAI_KEY", model="custom-model", timeout=12.5)

    assert observed == {"api_key": "sk-test-secret", "timeout": 12.5}


def test_openai_provider_diagnostics_do_not_contain_prompt_or_auth_secret(monkeypatch):
    secret = "sk-test-secret"
    prompt = "private prompt text"
    monkeypatch.setenv("TEST_OPENAI_KEY", secret)

    class ExplodingClient:
        def __init__(self, **kwargs):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise RuntimeError("transport failed")

    monkeypatch.setattr(
        "chiplet_layout_workflow.providers.openai_provider.OpenAI", ExplodingClient
    )

    provider = OpenAIProvider(api_key_env="TEST_OPENAI_KEY")
    with pytest.raises(RuntimeError) as error:
        provider.generate(prompt, "user secret", [], None)

    diagnostic = str(error.value)
    assert secret not in diagnostic
    assert prompt not in diagnostic


def test_openai_provider_includes_text_artifact_in_outgoing_request(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    captured = {}

    class Response:
        id = "req-1"
        model = "custom-model"
        usage = None
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]

    class Completions:
        def create(self, **request):
            captured.update(request)
            return Response()

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    artifact = tmp_path / "evidence.md"
    artifact.write_text("measured evidence", encoding="utf-8")

    OpenAIProvider(api_key_env="TEST_OPENAI_KEY").generate("system", "user", [artifact], None)

    outgoing_user = captured["messages"][1]["content"]
    assert f"--- ARTIFACT: {artifact} ---" in outgoing_user
    assert "measured evidence" in outgoing_user
    assert f"--- END ARTIFACT: {artifact} ---" in outgoing_user


def test_openai_provider_wraps_raw_schema_for_structured_output(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    captured = {}

    class Response:
        id = "req-1"
        model = "custom-model"
        usage = None
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]

    class Completions:
        def create(self, **request):
            captured.update(request)
            return Response()

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    schema = {"title": "Reader response / v1", "type": "object"}

    OpenAIProvider(api_key_env="TEST_OPENAI_KEY").generate(
        "system", "user", [], response_schema=schema
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Reader_response_v1",
            "strict": True,
            "schema": schema,
        },
    }


def test_openai_provider_rejects_oversize_artifact_before_request(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")

    class Completions:
        def create(self, **kwargs):
            raise AssertionError("request should not be called")

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    artifact = tmp_path / "large.txt"
    artifact.write_bytes(b"x" * (64 * 1024 + 1))
    provider = OpenAIProvider(api_key_env="TEST_OPENAI_KEY")

    with pytest.raises(ArtifactError, match="too large"):
        provider.generate("system", "user", [artifact], None)


def test_openai_provider_rejects_oversize_combined_artifacts_before_request(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    requests = []

    class Response:
        id = "req-1"
        model = "custom-model"
        usage = None
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return Response()

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    artifacts = []
    for index in range(17):
        artifact = tmp_path / f"part-{index}.txt"
        artifact.write_bytes(b"x" * (64 * 1024 - 1))
        artifacts.append(artifact)
    provider = OpenAIProvider(api_key_env="TEST_OPENAI_KEY")

    with pytest.raises(ArtifactError, match="combined artifacts are too large"):
        provider.generate("system", "user", artifacts, None)

    assert requests == []


def test_openai_provider_allows_combined_artifacts_at_exact_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    requests = []

    class Response:
        id = "req-1"
        model = "custom-model"
        usage = None
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return Response()

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    artifacts = []
    for index in range(16):
        artifact = tmp_path / f"exact-{index}.txt"
        artifact.write_bytes(b"x" * (64 * 1024))
        artifacts.append(artifact)
    provider = OpenAIProvider(api_key_env="TEST_OPENAI_KEY")

    provider.generate("system", "user", artifacts, None)

    assert len(requests) == 1


def test_openai_provider_closes_client(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    closed = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)
    provider = OpenAIProvider(api_key_env="TEST_OPENAI_KEY")

    provider.close()

    assert closed == [True]


def test_openai_provider_context_manager_closes_client(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-secret")
    closed = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr("chiplet_layout_workflow.providers.openai_provider.OpenAI", Client)

    with OpenAIProvider(api_key_env="TEST_OPENAI_KEY"):
        pass

    assert closed == [True]
