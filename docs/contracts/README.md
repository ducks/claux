# One-shot consumer contract

Run `claux --print '...' --output-format json` to receive one JSON result on
stdout. Diagnostics go to stderr. Redirect stdout to `result.json` if needed;
Claux does not create a file with that name automatically. Use `--transcript
PATH` to save a transcript (see `claux --help` for all supported options).

## Versions and compatibility

The result format is version 1; the transcript format is version 2. These are
independent contracts. Consumers must ignore unknown fields and accept additive
fields without requiring an exact schema-version match. Additive changes do not
increment the version. A breaking change requires a new version and migration
documentation. Do not blindly accept a future breaking version: consumers should
declare which versions they understand, separately for results and transcripts.

Fixtures in this directory are checked against a real Engine turn driven by a
deterministic provider. Timing values alone are normalized to zero. Run
`cargo test output::tests` to check the contract.

## Result

See [result.json](fixtures/result.json) for a completed result. Required fields:
`schema_version`, `model`, `result`, `usage`, and `outcome`.

On a turn failure, `result` is null and `outcome` is an error object:

```json
{"status":"error","message":"provider disconnected","failure":{"kind":"network","retryable":true,"attempts":1}}
```

`failure` may be absent when classification is unavailable. Its optional
`http_status` and `retry_after_ms` fields are omitted when unknown. `retryable`
describes the failure class, not a promise that further retries will succeed.
Early CLI/configuration/startup failures may occur before an engine exists and
are not guaranteed to emit a JSON result. Check the exit status and handle empty
stdout. Never infer failure classification by searching human-readable prose.

## Transcript and checkpoints

See [transcript.json](fixtures/transcript.json). In addition to model, usage, and
outcome, it contains `messages`, `tool_trace`, and `timing`. `archive`, when
present, contains original conversation events as `{id, message}` records and
can include history no longer in active `messages` after compaction.

`outcome.status` is `running`, `completed`, or `error`. A running checkpoint is
not a successful final result. Completed outcomes include `result`; errors use
the error shape above. The transcript is published by replacing the destination
with a completed temporary file. A crash may leave the last running checkpoint.

Messages have `role` and `content`. Content is a string or an array of tagged
blocks: `text`, `image`, `reasoning`, `tool_use`, or `tool_result`. Readers should
retain or gracefully omit unknown future blocks, not reject the entire file.
Images contain base64 data. Reasoning may contain opaque provider state.

Each tool trace entry has `id`, `name`, `input`, `output`, `is_error`, `read_only`,
`started_after_ms`, and `duration_ms`. A tool failure does not necessarily mean
the overall turn failed. `timing` includes `total_duration_ms` and `model_rounds`;
each model round has `index`, `started_after_ms`, `duration_ms`, `status`, and
optional `failure` and `usage`. Times are milliseconds, not wall-clock dates.

## Usage

Both formats use the same usage object:

- `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`:
  nonnegative cumulative counts. Input excludes the separately counted cache
  tokens. Missing provider usage cannot be reconstructed from zero.
- `cost_usd`: number when known, otherwise null. It may be provider-reported or
  estimated; it is not a billing guarantee.
- `cost_source`: `provider`, `estimated`, `mixed`, `unavailable`, or `incomplete`.
  Estimates are accumulated only for usage without a reported charge, at the
  pricing active for that request. Unpriced usage makes the total null.
  Built-in exact-ID prices are standard short-context estimates, not tiered
  billing calculations. Metadata was checked on 2026-10-03 against
  [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
  and [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
  Configured metadata and available catalog data override these fallbacks;
  unknown models use a conservative 128,000-token window and no price.
- `rounds`: turn-loop provider requests started, including retries, failed
  requests, and sub-agent rounds. Compaction requests are excluded.
- `tool_calls`: finalized tool results, including sub-agent, denied, failed, and
  interrupted calls. Running calls are not counted until finalized.

Transcripts include `sub_agents` when the Agent tool runs. Each report has its
`parent_tool_use_id`, usage, model rounds, and tool trace. Child timings are
relative to the child execution; the parent tool trace places that execution
on the parent timeline. Usage totals already include these reports, so consumers
must not add them again. Sub-agents inherit the parent's output-token limit and
rely on the parent's worktree checkpoint rather than capturing another one.

Counts share the usage reset boundary. In a new one-shot engine they cover that
invocation; interactive usage accumulates until reset. Sub-agent accounting is
not currently included in these totals.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Successful completion |
| 1 | Generic/unclassified failure or startup failure |
| 2 | CLI usage error |
| 10 | Cancellation |
| 11 | Rate limit or provider unavailable |
| 12 | Authentication/authorization/billing failure |
| 13 | Input context exceeded |
| 14 | Policy rejection |
| 15 | Protocol error or malformed tool arguments |
| 16 | Model not found |
| 17 | Network failure |
| 18 | Output token limit exceeded |

An externally killed process may instead have a platform-specific signal exit
status and only a running checkpoint. JSON consumers should inspect both the
process status and the outcome.

Configuration-file layout is not part of this JSON contract. Until dedicated
transport flags are available, generate TOML with a TOML parser rather than
depending on line order or editing it with sed.
