package rules

import (
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

type MissingSeverabilityRule struct{}

func (MissingSeverabilityRule) Code() string {
	return "B2B019"
}

func (MissingSeverabilityRule) Run(doc *model.Document, _ *model.Indexes) []model.Diagnostic {
	text := strings.ToLower(doc.Text)
	if strings.Contains(text, "severability") || strings.Contains(text, "invalid or unenforceable") {
		return nil
	}
	return []model.Diagnostic{{
		Code:     "B2B019",
		Severity: model.SeverityWarning,
		Message:  "Contract appears to lack a severability clause.",
		Suggestion: &model.Suggestion{
			Summary: "Add a standard severability clause in the miscellaneous section.",
			Pattern: "If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall remain in full force and effect.",
		},
	}}
}

type MissingGoverningLawOrJurisdictionRule struct{}

func (MissingGoverningLawOrJurisdictionRule) Code() string {
	return "B2B041"
}

func (MissingGoverningLawOrJurisdictionRule) Run(_ *model.Document, idx *model.Indexes) []model.Diagnostic {
	if idx.Dispute.HasGoverningLaw || idx.Dispute.HasCourtsAt || idx.Dispute.HasJurisdiction {
		return nil
	}
	return []model.Diagnostic{{
		Code:     "B2B041",
		Severity: model.SeverityWarning,
		Message:  "Contract appears to lack a governing law or jurisdiction clause.",
		Suggestion: &model.Suggestion{
			Summary: "Add a governing law and forum clause aligned to the intended Indian jurisdiction.",
			Pattern: "This Agreement shall be governed by and construed in accordance with the laws of India. Subject to the arbitration clause, the courts at [City] shall have exclusive jurisdiction.",
		},
	}}
}

type ConflictingDisputeResolutionRule struct{}

func (ConflictingDisputeResolutionRule) Code() string {
	return "B2B052"
}

func (ConflictingDisputeResolutionRule) Run(_ *model.Document, idx *model.Indexes) []model.Diagnostic {
	if !(idx.Dispute.HasArbitration && idx.Dispute.HasCourtsAt) {
		return nil
	}
	return []model.Diagnostic{{
		Code:     "B2B052",
		Severity: model.SeverityWarning,
		Message:  "Document contains both arbitration language and direct court-forum language. Review dispute-resolution consistency.",
		Suggestion: &model.Suggestion{
			Summary: "Clarify whether disputes go to arbitration first, and whether court jurisdiction is limited to interim relief or enforcement.",
			Pattern: "Any dispute arising out of or in connection with this Agreement shall be finally resolved by arbitration... Courts at [City] shall have jurisdiction only for interim relief and enforcement of the award.",
		},
	}}
}
