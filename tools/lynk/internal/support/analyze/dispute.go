package analyze

import (
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

func ExtractDisputeResolutionFacts(doc *model.Document) model.DisputeResolutionFacts {
	text := strings.ToLower(doc.Text)
	return model.DisputeResolutionFacts{
		HasArbitration:  strings.Contains(text, "arbitration"),
		HasCourtsAt:     strings.Contains(text, "courts at"),
		HasJurisdiction: strings.Contains(text, "jurisdiction"),
		HasGoverningLaw: strings.Contains(text, "governing law") || strings.Contains(text, "laws of india"),
	}
}
