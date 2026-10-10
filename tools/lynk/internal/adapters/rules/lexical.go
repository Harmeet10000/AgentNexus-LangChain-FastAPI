package rules

import (
	"regexp"
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

var dayPattern = regexp.MustCompile(`(?i)\b\d+\s+days\b`)

type AmbiguousTimeComputationRule struct{}

type InconsistentDefinitionCaseRule struct{}

func (AmbiguousTimeComputationRule) Code() string {
	return "LEX012"
}

func (InconsistentDefinitionCaseRule) Code() string {
	return "LEX005"
}

func (AmbiguousTimeComputationRule) Run(doc *model.Document, _ *model.Indexes) []model.Diagnostic {
	diagnostics := make([]model.Diagnostic, 0)
	for _, paragraph := range doc.Paragraphs {
		match := dayPattern.FindString(paragraph.Text)
		if match == "" {
			continue
		}
		lower := strings.ToLower(paragraph.Text)
		if strings.Contains(lower, "business days") || strings.Contains(lower, "calendar days") {
			continue
		}
		diagnostics = append(diagnostics, model.Diagnostic{
			Code:      "LEX012",
			Severity:  model.SeverityWarning,
			Message:   "Unqualified use of 'days'. Specify 'Business Days' or 'Calendar Days'.",
			Evidence:  paragraph.Text,
			Paragraph: paragraph.Index,
			Suggestion: &model.Suggestion{
				Summary:     "Replace the time period with a qualified unit.",
				Replacement: strings.Replace(paragraph.Text, match, strings.Replace(match, "days", "Business Days", 1), 1),
			},
		})
	}
	return diagnostics
}

func (InconsistentDefinitionCaseRule) Run(doc *model.Document, idx *model.Indexes) []model.Diagnostic {
	diagnostics := make([]model.Diagnostic, 0)
	for termKey, definition := range idx.Definitions {
		lowerTerm := strings.ToLower(definition.Term)
		for _, paragraph := range doc.Paragraphs {
			if paragraph.Index == definition.Paragraph {
				continue
			}
			textLower := strings.ToLower(paragraph.Text)
			if !strings.Contains(textLower, lowerTerm) {
				continue
			}
			if strings.Contains(paragraph.Text, definition.Term) {
				continue
			}
			diagnostics = append(diagnostics, model.Diagnostic{
				Code:      "LEX005",
				Severity:  model.SeverityWarning,
				Message:   "Defined term is used with inconsistent capitalization.",
				Evidence:  paragraph.Text,
				Paragraph: paragraph.Index,
				Suggestion: &model.Suggestion{
					Summary:     "Use the defined term with its original capitalization.",
					Replacement: strings.Replace(textLower, termKey, definition.Term, 1),
				},
			})
		}
	}
	return diagnostics
}
