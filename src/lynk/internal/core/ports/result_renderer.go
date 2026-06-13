package ports

import (
	"io"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

type ResultRenderer interface {
	Render(writer io.Writer, result model.LintResult) error
}
