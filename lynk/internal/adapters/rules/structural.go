package rules

import (
	"fmt"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

type UnresolvedCrossReferenceRule struct{}

func (UnresolvedCrossReferenceRule) Code() string {
	return "STR001"
}

func (UnresolvedCrossReferenceRule) Run(doc *model.Document, idx *model.Indexes) []model.Diagnostic {
	diagnostics := make([]model.Diagnostic, 0)
	for _, reference := range idx.References {
		if _, ok := idx.SectionsByNumber[reference.TargetSection]; ok {
			continue
		}

		paragraphText := ""
		for _, paragraph := range doc.Paragraphs {
			if paragraph.Index == reference.SourceParagraph {
				paragraphText = paragraph.Text
				break
			}
		}

		diagnostics = append(diagnostics, model.Diagnostic{
			Code:      "STR001",
			Severity:  model.SeverityError,
			Message:   fmt.Sprintf("Reference points to missing Section %s.", reference.TargetSection),
			Evidence:  paragraphText,
			Paragraph: reference.SourceParagraph,
			Suggestion: &model.Suggestion{
				Summary: "Review numbering and update the cross-reference to an existing section.",
			},
		})
	}
	return diagnostics
}
