# MCP limits

Each server accepts `timeout_seconds` in TOML or `.mcp.json` (default 120,
valid range 1 through 3600). Cancellation and timeout notify only the affected
request; they do not disconnect unrelated tools. A server may ignore cancellation
and side effects may already have occurred.

Server names and exposed names (`mcp__server__tool`) must contain only ASCII
letters, digits, underscores, and hyphens. Exposed names are limited to 64 bytes.
Duplicate tool names are rejected before registering any tools in the batch.

Tool descriptions are capped at 4096 bytes. Input schemas must describe an
object, be valid JSON Schema, and fit in 32 KiB. External file/network schema
references are not resolved. Invalid tools cause that server's discovery to fail
with a diagnostic rather than sending invalid definitions to the model.

Text results are capped at 64 KiB plus a truncation marker. Non-text content is
represented by an omission marker instead of forwarding base64 into context.
