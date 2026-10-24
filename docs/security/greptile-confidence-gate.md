# Greptile 5/5 confidence gate

`.github/workflows/greptile-confidence-gate.yml` is a trusted
`pull_request_target` workflow. It checks the current pull request through the
GitHub REST API and publishes the stable `Greptile 5/5 confidence gate`
check-run context against the exact pull-request head SHA.

The verifier accepts a review only when all of these conditions hold:

- The issue comment author is the Greptile GitHub App bot with the expected
  login, numeric user ID, and `Bot` type.
- The current comment body contains a confidence score, and every score in the
  comment is exactly `5/5`.
- The current comment body contains exactly one distinct `Last reviewed
  commit` SHA, equal to the current pull-request head.
- The pull-request head is unchanged between the initial metadata read and the
  final verification read.

The workflow checks the current comment representation. Deleted comments are
therefore absent, and edited comments must still satisfy the complete parser.
Old checks are attached to their old SHA and cannot satisfy a required check on
a later head. The workflow also cancels an in-progress run when a newer push
arrives for the same pull request.

The workflow uses only `contents: read`, `pull-requests: read`, and
`checks: write`. It checks out the default branch with credentials disabled and
never checks out or executes pull-request code. API failures, malformed
responses, timeouts, conflicting review text, missing reviews, and head races
fail closed. If the verifier cannot publish a failure check because GitHub is
unavailable, the required check remains absent/pending rather than passing.

To rerun manually, use **Actions → Greptile 5/5 confidence gate → Run
workflow** and enter the open pull-request number. The job has a five-minute
timeout; a timeout leaves the required check non-success and should be rerun
after the dependency recovers. The existing `Greptile Review` requirement
must remain in place until this check has completed successfully on a real PR
and its exact check-run integration has been verified.
