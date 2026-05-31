package output

import (
	"fmt"
	"io"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

type TextRenderer struct{}

func (TextRenderer) Render(writer io.Writer, result model.LintResult) error {
	if _, err := fmt.Fprintf(writer, "Lynk lint report: %s\n", result.DocumentTitle); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(writer, "Path: %s\n\n", result.DocumentPath); err != nil {
		return err
	}
	if len(result.Diagnostics) == 0 {
		_, err := fmt.Fprintln(writer, "No diagnostics found.")
		return err
	}
	for _, diagnostic := range result.Diagnostics {
		if _, err := fmt.Fprintf(writer, "[%s] %s: %s\n", diagnostic.Code, diagnostic.Severity, diagnostic.Message); err != nil {
			return err
		}
		if diagnostic.Evidence != "" {
			if _, err := fmt.Fprintf(writer, "  evidence: %s\n", diagnostic.Evidence); err != nil {
				return err
			}
		}
		if diagnostic.Suggestion != nil {
			if _, err := fmt.Fprintf(writer, "  suggestion: %s\n", diagnostic.Suggestion.Summary); err != nil {
				return err
			}
		}
	}
	return nil
}
