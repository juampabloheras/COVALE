import subprocess
import sys


def test_core_package_imports_without_provider_modules() -> None:
    code = """
import sys
for name in (
    "deepagents",
    "dotenv",
    "langchain_openai",
    "openai",
):
    sys.modules[name] = None
import covale
print(covale.COVALE.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "COVALE"
