package service

import (
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/ports"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/support/analyze"
)

type LintService struct {
	source ports.DocumentSource
	rules  []ports.Rule
}

func NewLintService(source ports.DocumentSource, rules []ports.Rule) *LintService {
	return &LintService{
		source: source,
		rules:  rules,
	}
}

type RuleFilter struct {
	IncludeCodes map[string]struct{}
	ExcludeCodes map[string]struct{}
}

func (service *LintService) Filtered(filter RuleFilter) *LintService {
	selected := make([]ports.Rule, 0, len(service.rules))
	for _, rule := range service.rules {
		code := strings.ToUpper(rule.Code())
		if len(filter.IncludeCodes) > 0 {
			if _, ok := filter.IncludeCodes[code]; !ok {
				continue
			}
		}
		if _, excluded := filter.ExcludeCodes[code]; excluded {
			continue
		}
		selected = append(selected, rule)
	}
	return NewLintService(service.source, selected)
}

func (service *LintService) Lint(path string) (model.LintResult, error) {
	doc, err := service.source.Read(path)
	if err != nil {
		return model.LintResult{}, err
	}

	indexes := analyze.BuildIndexes(doc)
	diagnostics := make([]model.Diagnostic, 0)
	for _, rule := range service.rules {
		diagnostics = append(diagnostics, rule.Run(doc, indexes)...)
	}

	return model.LintResult{
		DocumentTitle: doc.Title,
		DocumentPath:  doc.SourcePath,
		Diagnostics:   diagnostics,
	}, nil
}
