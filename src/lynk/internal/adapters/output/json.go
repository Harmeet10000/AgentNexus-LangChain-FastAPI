package output

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

type JSONRenderer struct{}

func (JSONRenderer) Render(writer io.Writer, result model.LintResult) error {
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		return fmt.Errorf("encode lint result json: %w", err)
	}
	return nil
}
