package analyze

import "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"

func BuildSectionIndex(doc *model.Document) (map[string]model.Section, []string) {
	sectionsByNumber := make(map[string]model.Section, len(doc.Sections))
	sectionOrder := make([]string, 0, len(doc.Sections))
	for _, section := range doc.Sections {
		if section.Number == "" {
			continue
		}
		sectionsByNumber[section.Number] = section
		sectionOrder = append(sectionOrder, section.Number)
	}
	return sectionsByNumber, sectionOrder
}
