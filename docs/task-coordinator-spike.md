# Task coordinator spike

Branch: `feature/task-coordinator-spike`. Extends the native `claux task`
commands with a durable inbox and optional structured worker reports. No
Scaledown, extra provider, Python wrapper, GitHub integration, or daemon is
required. Model profiles, session storage, permissions, and containment remain
those of ordinary Claux tasks.

## Try it

In the full-screen TUI, use `/task start <instructions>` to launch and select a
worker, or `/task start-edit <instructions>` to create a worktree with edit
permission. `/tasks` lists workers, `/task open <id>` selects one, and `/task
refresh` shows its status, report, and recent activity. Use `/task send
<instructions>` to queue a follow-up, and `/task resume` or `/task cancel` to
control it. Workers use the current session's model but do not receive its
conversation automatically, so include the context they need in the prompt.

The CLI provides the same task records and controls:

```sh
cargo build
./target/debug/claux task start 'Investigate the failing tests and implement a fix' \
  --cwd /absolute/path/to/repo --worktree --model my-profile \
  --structured-report --permission-mode accept-edits
./target/debug/claux task send TASK_ID 'Also check the regression case in parser.rs'
./target/debug/claux task status TASK_ID
./target/debug/claux task attach TASK_ID --follow
./target/debug/claux task wait TASK_ID
./target/debug/claux task resume TASK_ID
```

Use the exact task ID printed by `start`. `send` can now run while the worker
is active: it appends an input without interrupting the model request. The
worker handles queued messages one at a time, in order, with the saved
conversation from preceding turns. There is no mid-turn injection. A worker
initializes a fresh engine for each turn and reloads its pinned provider binding.
Background operations that would require permission prompts remain denied.
`accept-edits` allows edits under the existing policy; shell commands may still
need explicit rules. The coordinator grants no extra permissions.

A normal `waiting` report allows the next queued turn to run. Reports of
`needs_input`, `ready_for_review`, or `failed` hold the remaining inbox. Cancelled
or failed engine turns also stop draining. `resume` explicitly releases held
queued work; `send` to an inactive, non-interrupted task adds a new message and
releases its queue too. Existing queued messages retain FIFO order, including
messages queued before a question or cancellation. Inspect them before resuming.
Cancellation retains queued inputs; it does not delete them. There is no queue
editing command in this spike. A cancellation received before the next input is
claimed stops the worker without consuming that input.

## Reports and transport outcomes

`--structured-report` asks the worker to end with a JSON object:

```json
{"state":"needs_input","summary":"Both targets are available; no deployment attempted","question":"Which cluster should receive the change?"}
```

States are `waiting`, `needs_input`, `ready_for_review`, and `failed`.
`needs_input` requires a nonempty question. Invalid JSON, missing fields, or
unknown states hold the queue for human review rather than guessing from prose.
The contract is an additional system instruction, not a provider-specific
response-format feature, so any configured model can attempt it.

These reports are model claims, marked `advisory: true`. They are neither proof
of test success nor authorization to deploy, push, merge, or message anyone.
Without the option, ordinary output remains unchanged and a neutral `waiting`
report records only that the turn ended.

Status retains its existing process `status` and adds `stage`, the last `report`,
`queued_inputs`, and input IDs/phases without prompt bodies. The last report
can describe a previous turn while a new one is working. `wait` retains the
existing transport meaning: successful model completion can still have a
`needs_input` or `failed` task report. Inspect `task_report` in its JSON result.
The original failure classification and output are retained.

## Durability and recovery

Each task has an atomically replaced, synced `inbox.json`. Control operations
and queue transitions share `control.lock`; one worker owns `worker.lock`.
An input moves `queued → claimed → finished`. Claiming is persisted before
submitting it to the engine, and finishing follows session/result persistence.
The worker releases its lock under the control lock after its final queue check,
preventing a concurrent send from being stranded during normal shutdown.

A crash after a claim is deliberately ambiguous: the model may have edited
files or run commands. Status shows `interrupted`, and send/resume/retry refuse
to replay it. Inspect the checkpoint, session, worktree, and external effects.
After inspection, `claux task recover TASK_ID --abandon INPUT_ID` records the
exact claimed input as abandoned, without replaying it or starting queued work.
Use the input ID from `task status`, then use `task resume TASK_ID` separately
if the queued work is still appropriate. Recovery does not undo side effects or
reconcile the conversation; start a new task if the saved session is unsuitable.
Automated reconciliation remains future work.
A launch failure before any claim can use `task retry`. The queue survives
restart but does not start a worker automatically after an uncertain crash.
This provides at-most-once automatic dispatch, not exactly-once external effects.

Legacy `pending.json` is read only when no inbox exists. On migration, the
committed inbox becomes authoritative before the old pending file is removed.
Do not manage a migrated task with an older binary.

Each prompt is limited to 40,000 bytes, at most 32 inputs may be queued, and a
task accepts at most 256 lifetime receipts. Create a new task after that cap.
Each turn remains bounded by its task timeout and the existing event-log cap.
Task directories are private; inboxes, sessions, reports, and results may contain
private project text. Credentials continue to resolve through normal Claux
configuration. `result.json` and `report.json` describe the latest turn; immutable
`result-INPUT_ID.json` files retain completed per-input outcomes and reports.

## Validation and next steps

Unit tests exercise FIFO claims, interrupted-claim refusal, legacy migration,
and report validation. Local-provider integration tests queue messages while
an HTTP response is held open, verify conversation continuity, hold pending work
at review, resume explicitly, cancel a turn, and retry an unconsumed launch.
They do not use paid models.

The next layers would be source adapters (CLI events, CI, GitHub), deduplicated
external event IDs, explicit task scopes, and reconciliation of external writes.
Those are not implemented here. Scaledown could optionally classify or summarize
events later; scheduling and permissions would still be enforced by code.
