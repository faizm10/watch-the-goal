# Security policy

## Supported versions

Security fixes go into the latest release and `main`.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a vulnerability

Please don't open a public issue. Report it privately through
[GitHub security advisories](https://github.com/faizm10/watch-the-goal/security/advisories/new)
instead. Include what you found, how to reproduce it, and what it lets someone do.

You'll get a reply within a week. Once a fix is released, the advisory is
published with credit to you, unless you'd rather stay anonymous.

## How Watch the Goal handles your data

- The web server listens on `127.0.0.1` by default. It has no login, so don't
  use `--host` to expose it to a network you don't trust.
- Footage is read in place and never modified. Dropped files are copied into
  the workspace (`~/WatchTheGoal` by default).
- AI review only talks to loopback addresses (a local Ollama).
- The optional caption writer sends only text (team names, score, match minutes,
  tags and detector notes) to the provider you choose. It never sends footage,
  frames or file names. A pasted API key is held in memory for the session and
  never written to disk.
- Model downloads come from pinned revisions and are checked against a SHA-256
  before use.
