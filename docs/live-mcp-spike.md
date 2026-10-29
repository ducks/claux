# Live MCP spike

The TUI and REPL accept these owner-issued commands between turns:

```
/mcp list
/mcp load local-slug
/mcp unload local-slug
```

Claux can write a server and its project configuration using its existing file
tools. The owner then loads it explicitly. There is no model-callable management
tool, automatic promotion, tool-validation gate, or persistent learning loop yet.

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

MCP servers run as ordinary child processes with the existing sanitized
inherited environment and explicitly configured environment variables. They do
not inherit the Bash tool's filesystem containment. Inspect generated code and
configuration before loading it; project trust is not a per-server sandbox.

Unload removes that server's tools using explicit ownership metadata. Dropping
the last client requests asynchronous transport/process cleanup; the command
does not wait for proof of process exit. Current-session tool approvals are
cleared after a successful change. Conversation history is preserved, tool-token
estimates refreshed, and provider session cursors reset so the next request
includes the updated schema and complete history. Background agents already
running retain their own tool registries.
