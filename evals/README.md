# Agent behavior evaluations

`fixtures/agent_contracts.json` drives the real Claux turn loop with a
deterministic provider. The suite executes real tools inside temporary
workspaces and asserts:

- multi-round tool selection and ordering;
- resulting file contents;
- permission denials and error propagation;
- steering preemption;
- recovery from unavailable tools;
- rejection of incomplete provider streams.

Run it without credentials or network access:

```bash
cargo test evals::deterministic_agent_contracts -- --nocapture
```

Each fixture defines its initial files, scripted provider rounds, permission
mode, and observable expectations. `$ROOT` in tool inputs expands to that
scenario's isolated temporary workspace.

There is also an ignored, paid Anthropic smoke test:

```bash
ANTHROPIC_API_KEY=... \
  cargo test evals::live_anthropic_smoke -- --ignored --nocapture
```

The `Live Agent Evaluation` workflow exposes the same test through a manual
GitHub Actions dispatch. It never runs on ordinary pushes or pull requests.
