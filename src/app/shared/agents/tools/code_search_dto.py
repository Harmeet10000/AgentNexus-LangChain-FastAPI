from pydantic import BaseModel, ConfigDict, Field


class RipgrepSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
        slots=True,
    )

    regex_pattern: str = Field(
        ..., 
        description="The exact regex pattern to search for."
    )
    file_extension: str | None = Field(
        default=None, 
        description="Optional: Restrict search to specific extensions (e.g., 'py', 'ts')."
    )
    max_results: int = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Maximum number of matches to return."
    )


class ShadowZoektSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
        slots=True,
    )

    symbol_name: str = Field(
        ..., 
        description="The exact function, class, or variable name to search for."
    )
    repo_name: str | None = Field(
        default=None, 
        description="Optional: Target specific repository."
    )