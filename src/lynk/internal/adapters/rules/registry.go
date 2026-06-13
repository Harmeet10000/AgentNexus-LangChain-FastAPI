package rules

import "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/ports"

func DefaultRegistry() []ports.Rule {
	return []ports.Rule{
		UnresolvedCrossReferenceRule{},
		InconsistentDefinitionCaseRule{},
		AmbiguousTimeComputationRule{},
		MissingSeverabilityRule{},
		MissingGoverningLawOrJurisdictionRule{},
		ConflictingDisputeResolutionRule{},
	}
}
