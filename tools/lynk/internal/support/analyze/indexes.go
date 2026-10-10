package analyze

import "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"

func BuildIndexes(doc *model.Document) *model.Indexes {
	sectionsByNumber, sectionOrder := BuildSectionIndex(doc)
	definitions := ExtractDefinitions(doc)
	references := make([]model.ReferenceEdge, 0)
	for _, paragraph := range doc.Paragraphs {
		for _, reference := range ExtractSectionReferences(paragraph.Text) {
			references = append(references, model.ReferenceEdge{
				SourceParagraph: paragraph.Index,
				TargetSection:   reference,
			})
		}
	}

	indexes := &model.Indexes{
		SectionsByNumber: sectionsByNumber,
		SectionOrder:     sectionOrder,
		Definitions:      definitions,
		References:       references,
		Dispute:          ExtractDisputeResolutionFacts(doc),
	}
	return indexes
}
