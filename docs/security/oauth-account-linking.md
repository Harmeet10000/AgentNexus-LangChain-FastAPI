# OAuth account-linking contract

The public OAuth callback is a login operation. It may create a new account
when the provider subject has no owner and the normalized email is unused. It
must never attach a new provider subject to an existing local account merely
because the emails match. A provider subject that is already linked is allowed
to log in by subject ownership.

## Explicit linking

1. An authenticated stateful session starts `/auth/oauth/{provider}/link/authorize`.
   The session must be active, verified, at the current authorization revision,
   and no older than the recent-authentication window.
2. Redis stores a five-minute link intent keyed by a digest of the OAuth state.
   The value binds the local user, provider, session, authorization revision,
   creation/expiry timestamps, and later the provider subject and email.
3. The provider callback verifies the signed state and provider email, then
   stores a candidate. It returns an opaque action token to the confirmation UI;
   it does not mutate the account or issue a login session.
4. The UI reads the candidate through the authenticated preview endpoint and
   displays the provider identity. A separate explicit confirmation consumes
   the action token atomically, revalidates the local session and authorization
   revision, rechecks provider-subject ownership, and only then links the
   identity.

The action token is stored only as a digest. `GETDEL` makes the first
confirmation win; replays, callback replays after completion, wrong sessions,
expired intents, and stale authorization revisions fail closed.

## Unlinking

Unlinking requires a recent stateful session and refuses to remove the final
usable sign-in method. Password-backed accounts may remove their OAuth method;
OAuth-only accounts must retain another provider before one can be removed.

## Migration and rollback

The compound unique index on `(oauth_accounts.provider,
oauth_accounts.provider_user_id)` must be created only after an operator has
run a duplicate report. Existing duplicate identities are ambiguous and must
be manually reconciled to one owner before the index is enabled. Existing
links are not detached automatically.

To roll back the feature, disable the link-start and unlink routes while
leaving the provider-subject login path and existing links intact. Pending
Redis intents expire naturally. Do not restore email-based auto-linking.

