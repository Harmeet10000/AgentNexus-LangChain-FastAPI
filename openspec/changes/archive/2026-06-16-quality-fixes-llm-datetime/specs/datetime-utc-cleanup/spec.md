# Capability: datetime-utc-cleanup

## Purpose

Replace deprecated `datetime.utcnow()` calls with `datetime.now(UTC)` for Python 3.12+ compliance.

## ADDED Requirements

### Requirement: Replace All datetime.utcnow() Calls

The codebase SHALL replace all `datetime.utcnow()` calls with `datetime.now(UTC)` using `UTC` from the `datetime` module. Affected files: `auth/service.py`, `auth/repository.py`, `users/repository.py`, `document_processing/models.py`.

#### Scenario: datetime.utcnow() replaced in auth module

- Given `auth/service.py` and `auth/repository.py`
- When the files are loaded
- Then no `datetime.utcnow()` calls exist
- And `datetime.now(UTC)` is used instead

#### Scenario: datetime.utcnow() replaced in users module

- Given `users/repository.py`
- When the file is loaded
- Then no `datetime.utcnow()` calls exist

#### Scenario: datetime.utcnow() replaced in document processing

- Given `document_processing/models.py`
- When the file is loaded
- Then default_factory lambdas use `datetime.now(UTC)`

## Non-Goals

- Timezone-aware datetime validation
- datetime library migration (pendulum, arrow, etc.)
