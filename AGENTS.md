# Project constraints

## Dataset intake idempotency

Before `ingest_local`, `download`, or any future dataset-intake action performs
provider/network work, creates an incoming workspace, reads dataset contents,
moves files, or records a job, it must check for both canonical-name and
canonical-URL/path conflicts. A conflict must abort the action and report the
existing dataset and storage path; duplicate datasets must never be processed
again. Repeat this check while holding the registry intake lock so concurrent
requests cannot bypass it. New intake routes must reuse the shared service
preflight rather than implementing their own duplicate handling.

### Process lock

If a user is running a download from a url, the same process should NOT be started the same canonical-url/path nor another download with the same canonical-name again by another user.

## Registered dataset protection

NEVER remove registered neural datasets nor any field in their meta-data (e.g., `url`). 

Field update:

- If future versions add new fields to the dataset meta (SQL), it should be updated automatically for old neural dataset entries.
- If future version deprecated any field, the field values of previously registered neural datasets should be preserved in the database, but not shown to user.

