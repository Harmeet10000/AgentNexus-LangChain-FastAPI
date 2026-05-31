package analyze

import "regexp"

var sectionReferencePattern = regexp.MustCompile(`(?i)section\s+(\d+(?:\.\d+)*)`)

func ExtractSectionReferences(text string) []string {
	matches := sectionReferencePattern.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	references := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		references = append(references, match[1])
	}
	return references
}
