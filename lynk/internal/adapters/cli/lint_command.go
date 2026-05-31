package cli

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/adapters/docx"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/adapters/output"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/adapters/rules"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/ports"
	coreservice "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/service"
)

func RunLint(args []string) int {
	flags := flag.NewFlagSet("lint", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	format := flags.String("format", "text", "output format: text or json")
	include := flags.String("include", "", "comma-separated rule codes to include")
	exclude := flags.String("exclude", "", "comma-separated rule codes to exclude")
	failOn := flags.String("fail-on", "", "exit non-zero when diagnostics at or above severity are present: warning or error")
	if err := flags.Parse(args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if flags.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: lynk lint [--format text|json] <document.docx>")
		return 2
	}

	service := coreservice.NewLintService(
		docx.Reader{},
		rules.DefaultRegistry(),
	).Filtered(coreservice.RuleFilter{
		IncludeCodes: parseRuleSet(*include),
		ExcludeCodes: parseRuleSet(*exclude),
	})
	result, err := service.Lint(flags.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "lint failed: %v\n", err)
		return 1
	}

	renderer := selectRenderer(*format)
	if renderer == nil {
		fmt.Fprintf(os.Stderr, "unsupported format: %s\n", *format)
		return 2
	}

	if err := renderer.Render(os.Stdout, result); err != nil {
		fmt.Fprintf(os.Stderr, "render output: %v\n", err)
		return 1
	}

	if shouldFail(result.Diagnostics, *failOn) {
		return 1
	}

	return 0
}

func selectRenderer(format string) ports.ResultRenderer {
	switch format {
	case "json":
		return output.JSONRenderer{}
	case "text":
		return output.TextRenderer{}
	default:
		return nil
	}
}

func parseRuleSet(raw string) map[string]struct{} {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	values := make(map[string]struct{})
	for _, part := range strings.Split(raw, ",") {
		trimmed := strings.ToUpper(strings.TrimSpace(part))
		if trimmed == "" {
			continue
		}
		values[trimmed] = struct{}{}
	}
	return values
}

func shouldFail(diagnostics []model.Diagnostic, failOn string) bool {
	threshold := strings.ToLower(strings.TrimSpace(failOn))
	if threshold == "" {
		return false
	}
	for _, diagnostic := range diagnostics {
		switch threshold {
		case "warning":
			if diagnostic.Severity == model.SeverityWarning || diagnostic.Severity == model.SeverityError {
				return true
			}
		case "error":
			if diagnostic.Severity == model.SeverityError {
				return true
			}
		}
	}
	return false
}
