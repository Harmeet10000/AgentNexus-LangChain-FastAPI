## Description
<!--- Describe your changes in detail -->

## Related Issue
<!--- Please link to the issue here -->
Closes #

## Motivation and Context
<!--- Why is this change required? What problem does it solve? -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup or refactor
- [ ] Dependency update
- [ ] CI/CD or build process changes

## How Has This Been Tested?
<!--- Please describe the tests you ran to verify your changes -->
- [ ] Unit tests (`pytest tests/unit`)
- [ ] Integration tests (`pytest tests/integration`)
- [ ] Manual testing
- [ ] Type checking (`mypy src`)
- [ ] Linting (`ruff check .`)
- [ ] Formatting (`ruff format --check .`)

## Test Instructions
<!--- Please describe how reviewers can test your changes -->
```bash
# Example:
# 1. Install dependencies: uv pip install -e ".[dev]"
# 2. Run tests: pytest
# 3. Start server: uvicorn src.main:app --reload
```

## Checklist
- [ ] My code follows the project's code style (Ruff formatting)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have added type hints to all function parameters and return types
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have updated the OpenAPI/Swagger documentation if API changes were made

## API Changes (if applicable)
<!--- Document any API endpoint changes, new routes, or modified request/response schemas -->
- [ ] No API changes
- [ ] New endpoints added
- [ ] Existing endpoints modified
- [ ] Breaking API changes (requires version bump)

## Database Changes (if applicable)
<!--- Document any database schema changes or migrations -->
- [ ] No database changes
- [ ] New migration added (`alembic revision`)
- [ ] Migration tested locally
- [ ] Rollback tested

## Performance Impact
<!--- Describe any performance implications of your changes -->
- [ ] No performance impact
- [ ] Performance improvement (describe below)
- [ ] Potential performance degradation (describe mitigation below)

## Security Considerations
<!--- Describe any security implications of your changes -->
- [ ] No security impact
- [ ] Security improvement
- [ ] Requires security review

## Screenshots (if appropriate)
<!--- Add screenshots for UI changes or API response examples -->

## Additional Notes
<!--- Add any additional notes or context about the pull request here -->
