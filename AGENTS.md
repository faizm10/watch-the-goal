# Watch the Goal

Use `uv run pytest -q` for the test suite. For changes to the web UI, follow [.agents/skills/webapp-testing/SKILL.md](.agents/skills/webapp-testing/SKILL.md) and use a temporary workspace for browser testing.

## Pull requests

- Open pull requests against `main` with a short descriptive title. Once GitHub assigns the PR number, rename the PR to `(#<PR number>) : <short description>`, for example `(#42) : Refine the video editor`. Keep the `#` so GitHub links the number to the pull request.
- Use the same format for the merge commit or squash commit subject, with the real PR number, so the commit on `main` links back to its pull request. Pass it explicitly when merging (for example `gh pr merge 42 --squash --subject "(#42) : Refine the video editor"`) rather than relying on GitHub's default subject. When asked to merge automatically, enable GitHub auto-merge if required checks or reviews are pending, and verify the merge status. Respect branch protection and required reviews.
