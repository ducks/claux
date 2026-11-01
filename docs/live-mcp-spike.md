# Live MCP spike

The TUI and REPL accept these owner-issued commands between turns:

```
/mcp list
/mcp load local-slug
/mcp unload local-slug
```

The model can also write and load tools itself during a conversation:

1. Write a stdio MCP server using `Write`/`Edit`, normally in
   `.claux/tools/<server-name>/server.py`.
2. Add its definition to the trusted project's `.mcp.json`.
3. Call `McpManage` with `{"action":"load","name":"local-slug"}`.
4. Call a discovered tool in the next round, such as `mcp__local-slug__slugify`.
5. Use `{"action":"unload","name":"local-slug"}`, edit the implementation,
   and load again to replace it. `{"action":"list"}` lists connected servers.

Generated servers default to `.claux/tools/<server-name>/` under the project
where Claux started. Override the base directory in the global `config.toml`,
or in a trusted project's `.claux.toml`:

```toml
generated_mcp_dir = ".claux/tools" # or "dev/agent-tools", or an absolute path
```

Relative paths resolve from the captured project directory; absolute paths are
used directly. `McpManage` includes the resolved default in its model-visible
description. This is a default convention, not a restriction on existing server
locations: the user can request another location. Directories are created by
normal file writes when needed. Choosing a directory does not grant permission
to write there or relax containment. `.mcp.json` remains at the project root.

`McpManage` follows normal mutation permissions and permission hooks. One-shot
sessions deny requests requiring confirmation. Failed loads return paired tool
errors, preserve the registry, and allow the model to repair its files and retry.
Discovery validates protocol/schema shape; it does not establish that a tool's
implementation is correct. This is runtime tool authoring, not weight learning
or an automatic, persistent promotion/learning loop. Files persist normally;
connection and approval changes are scoped to the running session.

To try it, start Claux in a trusted scratch project. Copy
`examples/live-mcp/server.py` there and add this `.mcp.json` **after startup**:

```json
{
  "mcpServers": {
    "local-slug": {
      "command": "python3",
      "args": ["/absolute/path/to/scratch/server.py"]
    }
  }
}
```

Run `/mcp load local-slug`, then ask for `mcp__local-slug__slugify` on
`Hello Live Tools`. Expected output: `hello-live-tools`. Unload the server,
edit it, and load again to discover the updated tools. Existing startup loading
still applies when configuration is already present at launch.

Project configuration is re-read from the original project directory, using
its captured trust decision. Untrusted projects cannot supply servers. Global
MCP definitions use the startup snapshot. Duplicate configured names, already
loaded servers, empty tool sets, and more than 16 connected tool-bearing servers
are rejected. Discovery has a 15-second deadline; existing per-call deadlines
still apply. Failed discovery leaves the registered tool set unchanged.

Owner-loaded and startup MCP servers retain their existing ordinary-process
launch behavior with sanitized inherited environment and explicitly configured
environment variables. Model-loaded servers instead use the session's Bash
command sandbox, including its configured filesystem policy. They never retry
an enforced sandbox failure unrestricted. An unrestricted Bash policy also means
unrestricted model-loaded servers; this does not add network isolation.

Model-loaded servers must omit `env`: passing arbitrary loader environment
variables to the containment helper would cross the enforcement boundary.
Arguments are shell-quoted as literals. Their discovered tools are treated as
mutating regardless of `readOnlyHint`, so generated annotations cannot grant
auto-approval. Project configuration still requires the captured project trust.
Loading runs the configured program, including initialization side effects.

Unload removes that server's tools using explicit ownership metadata. Dropping
the last client requests asynchronous transport/process cleanup; the command
does not wait for proof of process exit. Current-session tool approvals are
cleared after a successful change. Conversation history is preserved, tool-token
estimates refreshed, and provider session cursors reset so the next request
includes the updated schema and complete history. Background agents already
running retain their own tool registries.

The deterministic local workflow test uses a scripted provider and a real Python
stdio MCP process. It exercises file creation, repair after a missing definition,
schema refresh across rounds, invocation, unload, and provider cursor resets:

```sh
cargo test model_writes_loads_calls -- --ignored --nocapture
```

This verifies runtime plumbing, not a live model's ability to invent useful tools.
MCP image results remain unsupported by this spike.

A Linux CLI smoke uses a local HTTP provider fixture and exercises the actual
containment helper at server initialization and tool invocation. It also checks
literal quoting of server paths, dynamic schemas, and unload:

```sh
cargo build
python3 examples/live-mcp/smoke.py
```

The inherited Bash policy permits its normal temporary directories as well as
the workspace. The smoke's denied target is outside those allowed locations.
