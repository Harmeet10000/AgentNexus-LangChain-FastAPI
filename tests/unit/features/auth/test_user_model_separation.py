"""The auth aggregate is pure while Mongo retains its deployed schema."""

from pathlib import Path

from beanie import Document

from app.features.auth.adapters.persistence.mongodb.models import UserDocument
from app.features.auth.domain.user import User


def test_domain_user_is_not_a_beanie_document() -> None:
    assert not issubclass(User, Document)
    source = Path("src/app/features/auth/domain/user.py").read_text(encoding="utf-8")
    assert "from beanie" not in source
    assert "from pymongo" not in source


def test_mongo_document_preserves_collection_and_domain_field_parity() -> None:
    assert UserDocument.Settings.name == "users"
    assert set(UserDocument.model_fields) - {"revision_id"} == set(User.model_fields)

    index_names = {tuple(index.document["key"].keys()) for index in UserDocument.Settings.indexes}
    assert ("email",) in index_names
    assert ("oauth_accounts.provider", "oauth_accounts.provider_user_id") in index_names
    assert ("reset_token_hash",) in index_names
    assert ("verification_token_hash",) in index_names
