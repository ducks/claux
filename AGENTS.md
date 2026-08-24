# Repository instructions

## Workflow

- Work on a feature branch. Do not commit directly to `main`.
- Preserve unrelated local changes, session databases, credentials, and build
  artifacts.
- Use merge commits (`git merge --no-ff`) when a completed feature branch is
  merged into `main`.
- When resolving a GitHub issue, include `Fixes #<number>` in the feature
  commit so GitHub closes it when the commit reaches `main`.
- Do not push, publish to crates.io, create a release or tag, or merge into
  `main` unless the user explicitly asks.

## Project map

- `src/main.rs` wires CLI commands and starts interactive or one-shot sessions.
- `src/cli.rs` owns top-level command-line parsing and parse tests.
- `src/api/` owns provider protocols, streaming, usage, and structured provider
  errors. Keep generic OpenAI-compatible behavior generic.
- `src/config.rs` owns provider/model profiles, credential resolution, trust,
  and compatibility with legacy configuration.
- `src/query.rs` owns model/tool turn orchestration, retries, cancellation, and
  compaction decisions.
- `src/context.rs` contains the native Claux system prompt and context assembly.
- `src/tools/` owns native tools and their filesystem/permission behavior.
- `src/tui/` and `src/repl.rs` own the two interactive surfaces. Keep shared
  behavior consistent between them where applicable.
- `src/output.rs` and `src/checkpoint.rs` own stable one-shot/transcript output
  and recoverable turn checkpoints.
- `docs/screens/` contains tuishot-generated TUI screenshots checked by tests.

## Behavioral contracts

- Do not add model-specific engine behavior when a provider capability,
  protocol option, profile setting, or response shape can express it. Model
  IDs and pricing change frequently; provider-reported usage and configuration
  are the source of truth when available.
- Preserve valid conversation history across cancellation, failures,
  compaction, steering, and session resume. Tool uses must always have paired
  tool results before the next model request.
- Dropping or cancelling a provider stream must stop its HTTP reader so no
  detached request continues consuming tokens.
- Keep credentials out of configuration templates, sessions, transcripts,
  logs, debug output, and error bodies. Saved sessions contain credential-free
  transport bindings and resolve current credentials when reopened.
- Permission prompts and filesystem containment are separate boundaries.
  Approval must not disable containment, and a sandbox failure must never be
  retried unrestricted.
- Untrusted project configuration may tighten global permission, Bash, and
  native-tool policies, but may not loosen them or load project MCP servers.
- Preserve provider and failure classifications consumed by external harnesses
  such as Replaybook. Do not turn an evaluated model failure into success or an
  unavailable result merely to improve compatibility.
- Keep Linux-only containment behind platform gates. Provider, config, CLI,
  session, and TUI changes must continue to compile on Linux, macOS, and
  Windows.

## Testing and validation

- Add focused tests with each behavioral change. Prefer deterministic fixtures
  or local mock servers; live paid-provider tests must remain ignored or be
  explicitly invoked.
- While iterating, run the narrowest relevant test. Before handoff, run:

  ```sh
  make lint
  ```

  This checks formatting, runs Clippy with warnings denied, and runs the full
  test suite.
- Changes to TUI rendering may update tuishot snapshots. Review visual diffs in
  `docs/screens/`; do not accept regenerated snapshots blindly.
- Changes to sandbox behavior should run the Linux integration tests in
  `tests/bash_sandbox.rs` in addition to focused unit tests.
- Maintain the documented Rust 1.88+ compatibility unless the project
  explicitly raises its minimum supported Rust version.

## Configuration and API compatibility

- Preserve additive configuration behavior and legacy single-provider config
  unless a migration is explicitly requested.
- Keep OpenAI Chat Completions, OpenAI Responses, Anthropic, and compatible
  endpoints separated at the protocol layer. Do not assume every
  OpenAI-compatible provider supports every OpenAI parameter.
- Treat provider usage and cost fields as optional. Prefer provider-reported
  values over local estimates, but keep useful output when providers omit them.
- Bound network requests, subprocesses, captured output, retries, and
  concurrency. Surface an actionable error after the bound is exhausted.

## Releases

- Versions are date-based and may have multiple same-day patch releases.
- `make release` performs the version branch, merge commit, tag, push, and
  crates.io publication. It is externally mutating and must only run after the
  user explicitly requests a release.
- A pushed `v*` tag triggers cross-platform release binaries through GitHub
  Actions. Verify `make lint` and review the pending version before releasing.
