#!/usr/bin/env bash
# Checks that SecretStr fields are accessed via .get_secret_value(), not str() coercion.
# Exit 0 = clean, Exit 1 = violations found.

set -euo pipefail

SECRET_FIELDS=(
    JWT_SECRET_KEY
    NEO4J_PASSWORD
    GEMINI_API_KEY
    RESEND_API_KEY
    OAUTH_STATE_SECRET
    S3_ACCESS_KEY_ID
    S3_SECRET_ACCESS_KEY
    TAVILY_API_KEY
    PINECONE_API_KEY
)

violations=0

for field in "${SECRET_FIELDS[@]}"; do
    # Match str(settings.FIELD) or str(...settings.FIELD...) patterns
    matches=$(rg -n "str\(.*settings\.${field}" src/ -g '*.py' 2>/dev/null || true)
    if [ -n "$matches" ]; then
        echo "VIOLATION: str() coercion on ${field}:"
        echo "$matches"
        violations=$((violations + 1))
    fi
done

if [ "$violations" -gt 0 ]; then
    echo ""
    echo "Found ${violations} SecretStr str() coercion violations."
    echo "Fix: replace str(settings.FIELD) with settings.FIELD.get_secret_value()"
    exit 1
fi

echo "OK: no SecretStr str() coercions found."
