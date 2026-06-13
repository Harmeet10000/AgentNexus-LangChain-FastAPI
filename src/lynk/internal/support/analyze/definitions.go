package analyze

import (
	"regexp"
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

var definitionPattern = regexp.MustCompile(`"([A-Za-z][A-Za-z\s&/-]{1,80})"\s+means\b`)

func ExtractDefinitions(doc *model.Document) map[string]model.Definition {
	definitions := make(map[string]model.Definition)
	for _, paragraph := range doc.Paragraphs {
		matches := definitionPattern.FindAllStringSubmatch(paragraph.Text, -1)
		for _, match := range matches {
			if len(match) < 2 {
				continue
			}
			term := strings.TrimSpace(match[1])
			if term == "" {
				continue
			}
			definitions[strings.ToLower(term)] = model.Definition{
				Term:      term,
				Paragraph: paragraph.Index,
			}
		}
	}
	return definitions
}
