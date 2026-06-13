package model

type Definition struct {
	Term      string
	Paragraph int
	Section   string
}

type ReferenceEdge struct {
	SourceParagraph int
	TargetSection   string
}

type DisputeResolutionFacts struct {
	HasArbitration  bool
	HasCourtsAt     bool
	HasJurisdiction bool
	HasGoverningLaw bool
}

type Indexes struct {
	SectionsByNumber map[string]Section
	SectionOrder     []string
	Definitions      map[string]Definition
	References       []ReferenceEdge
	Dispute          DisputeResolutionFacts
}
