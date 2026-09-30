# Publishing

The [package publishing workflow](../.github/workflows/workflow.yml) builds the
source and wheel distributions, validates them with Twine, stores them as a
workflow artifact, and publishes them to PyPI.

## Configure trusted publishing

The workflow uses PyPI trusted publishing, so it does not require an API token.

1. Create a GitHub environment named `pypi` under **Settings > Environments**.
2. In the PyPI publishing settings for `covale`, add a trusted publisher with:
   - the GitHub organization or username that owns the repository;
   - the repository name;
   - workflow filename `workflow.yml`;
   - environment name `pypi`.

PyPI also supports a pending trusted publisher when the project has not been
published yet.

## Publish a release

1. Update `version` in [`pyproject.toml`](../pyproject.toml).
2. Commit and push the version change.
3. Create and publish a GitHub release for that commit.

Publishing the release runs the workflow automatically. It can also be started
manually from the repository's **Actions** tab.

[Back to the main README](../README.md#additional-resources)
