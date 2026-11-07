# Bounded filesystem searches

Glob and Grep run filesystem work on a blocking worker rather than the async
runtime thread. Both check cancellation and a cooperative five-second budget
between filesystem operations. Dropping the calling future cancels its worker's
child token too; a dropped async request does not intentionally leave a complete
recursive search running.

The defaults are fixed implementation limits for now:

| Limit | Glob | Grep |
| --- | --- | --- |
| Traversal/search time | 5 seconds | 5 seconds |
| Examined entries | 20,000 | 20,000 |
| Returned rows | 1,000 | 1,000 |
| Result text | 256 KiB | 256 KiB |
| Recursive depth | 64 | 64 |
| Individual file size/read | N/A | 8 MiB |
| Total bytes read | N/A | 64 MiB |

Small fixed notices may be appended beyond the result-text limit. Traversal counts
entries even when their names do not match. This bounds broad searches that return
few or no results, rather than limiting only successful matches.

Limits return the results collected so far with an explicit incomplete/truncated
notice and guidance to narrow the path or pattern. An empty incomplete search is
not reported as a complete “No matches found.” User cancellation is an error result,
with any collected results labelled partial. The tool input schema is unchanged;
models cannot increase these limits through tool arguments.

## Glob

Glob prunes hidden/build directories before descending, retaining the existing
include flags and explicit-base behavior. Literal pattern prefixes become the
walk root, and nonrecursive patterns do not walk unrelated descendants. Recursive
patterns do not follow directory symlinks. Matches within the examined scope are
sorted by modification time, most recent first; ties sort by path. A truncated
sample is not a guarantee of the newest matches in the entire tree.

Start at a relevant directory, for example `/etc/nginx` or the application's source
root, rather than `Glob {"pattern":"**/*","path":"/"}`. Root-wide searches now
stop at the same limits as any other search and return narrowing guidance.

## Grep

Grep stops for text output limits, traversal limits, read-byte limits, cancellation,
or elapsed time. Regular files larger than 8 MiB are skipped and reported as an
incomplete search. Special files such as FIFOs, devices, and sockets are not opened.
File reads and match callbacks check the same budget; long nonmatching files and
unterminated lines cannot bypass the byte bounds. Binary detection avoids scanning
binary data as ordinary text.

`files_with_matches` stops at the first match in each file. `count` scans all matching
lines within the budgets, fixing the previous behavior that stopped counting after
the first match. Counts in an interrupted/truncated search describe only the examined
content; the partial-results notice must not be omitted. Invalid glob filters or
output modes fail visibly rather than silently searching with a different filter.

## Boundaries

Read authorization is still checked for the base and each candidate. Restricted
search patterns cannot escape their base, denied directories are not descended,
and there is no unrestricted retry. This change does not alter Bash permissions,
MCP behavior, or tool-result pairing in conversation history.

The time limit is cooperative. A single OS operation blocked by an unavailable
filesystem cannot be forcibly interrupted by this implementation. The worker checks
its budget as soon as that operation returns. The change bounds normal local
traversal, reading, output and CPU work; it does not claim hard preemption of kernel
I/O. It adds no paid model tests and does not change Replaybook's frozen harness or
rescore prior runs.

## Validation

Deterministic fixtures cover no-match entry limits, result/byte/depth limits,
pre-cancellation, cancellation during reads, zero-duration deadlines, modification
ordering, literal-prefix and nonrecursive matching, hidden/build filtering,
multiple matching-line counts, oversized files, invalid filters, directory-symlink
containment and special-file skipping. Run `make lint` for formatting, Clippy and
all tests; the existing sandbox integration tests remain part of that suite.
