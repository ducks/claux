# Structured task reports

Branch: feature/structured-task-report, based on main. This feature is independent
of the task-automation spike and introduces no acceptance runner, publication or
new permissions.

## Usage

```sh
claux task start 'Investigate the failing tests and report your findings' \
  --cwd /path/to/repo --structured-report
```

Model-launched read-only research workers also receive the tool. Ordinary chat,
unstructured managed tasks, and lightweight Agent subagents do not. Narration
is normal assistant text; the worker finishes its handoff using TaskReport:

```json
{"state":"needs_input","summary":"Two deployment targets are configured","question":"Which target should receive the change?"}
```

Allowed states: waiting, needs_input, ready_for_review, failed. Summary must be
nonblank and at most 8,000 UTF-8 bytes. Question is optional (string/null), at most
2,000 bytes, and must be nonblank for needs_input. Unknown fields are rejected,
including acceptance_checks_passed, advisory and any other claimed verification
or authorization fields. No coercion or inference from prose is performed.

A report must be the only tool call in its batch, after all other work. Invalid
arguments or a mixed batch produce a paired tool error; the model can correct
its call. A successful call records a turn-local report and ends the model loop
without another provider round. The assistant tool call and corresponding result
are retained in valid conversation history for follow-ups. Tool traces and saved
sessions contain the report as ordinary tool data and may include private text.

## Authority and failure handling

TaskReport records the model's requested handoff. The coordinator adds
advisory=true and persists the normal report.json and per-input result artifacts.
A ready_for_review report does not mean tests passed, authorize broader work or
permit merge/push/deploy. Any later verification mechanism must record its own
evidence separately; the model cannot supply verification flags to this tool.

The tool changes in-memory session state, like TodoWrite. It uses normal tool
permissions/hooks, is available to read-only workers, and grants no filesystem
or command access. Explicit deny rules still apply.

Reports are cleared before each turn and before a new tool batch. Cancelled
calls do not record a handoff; cancelled/failed engine execution takes precedence
over a previously recorded report. Session reset removes stale state. A worker
that simply ends its answer without TaskReport gets needs_input with
missing_report=true. The coordinator never treats printed JSON as a fallback.

This changes the structured-report protocol for existing task records as well:
on the next turn their updated instructions/tool catalog request TaskReport.
The CLI flag, persisted report state names and status/attention queue fields stay
the same. Unstructured tasks keep their neutral waiting report and prose output.

## Verification

Unit tests cover state/length/question validation, rejection of forged evidence,
standalone calls, cancellation and reset. Local-provider integration tests cover
narration alongside tool reports, a rejected report followed by a corrected call,
read-only worker exposure, ordinary-chat absence, paired history across queued
follow-ups, review holds and cancellation. The number of provider requests also
checks that a valid report ends the turn without a redundant model round.

Run make lint for formatting, Clippy and all tests. These are deterministic
fixtures; no new live paid-model test is claimed for this branch.
