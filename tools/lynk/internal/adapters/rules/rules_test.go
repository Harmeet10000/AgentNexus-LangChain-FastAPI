package rules

import (
	"testing"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/adapters/docx"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
	"github.com/harmeet/langchain-fastapi-production/lynk/internal/support/analyze"
)

func TestRules(t *testing.T) {
	tests := []struct {
		name     string
		ruleCode string
		run      func(*model.Document, *model.Indexes) []model.Diagnostic
		doc      *model.Document
		wantCode string
	}{
		{
			name:     "missing severability emits warning",
			ruleCode: "B2B019",
			run:      MissingSeverabilityRule{}.Run,
			doc: &model.Document{
				Title:      "MSA",
				SourcePath: "msa.docx",
				Text:       "Payment terms only.",
			},
			wantCode: "B2B019",
		},
		{
			name:     "definition capitalization emits warning",
			ruleCode: "LEX005",
			run:      InconsistentDefinitionCaseRule{}.Run,
			doc: &model.Document{
				Title:      "MSA",
				SourcePath: "msa.docx",
				Text:       "\"Services\" means the services in the statement of work. the services shall be performed with care.",
				Paragraphs: []model.Paragraph{
					{Index: 0, Text: "\"Services\" means the services in the statement of work."},
					{Index: 1, Text: "the services shall be performed with care."},
				},
			},
			wantCode: "LEX005",
		},
		{
			name:     "unresolved reference emits error",
			ruleCode: "STR001",
			run:      UnresolvedCrossReferenceRule{}.Run,
			doc: &model.Document{
				Title:      "NDA",
				SourcePath: "nda.docx",
				Text:       "1. Confidentiality\n\nSubject to Section 9.",
				Paragraphs: []model.Paragraph{
					{Index: 0, Text: "1. Confidentiality", HeadingLvl: 1},
					{Index: 1, Text: "Subject to Section 9."},
				},
				Sections: []model.Section{{Index: 0, Number: "1", Title: "Confidentiality"}},
			},
			wantCode: "STR001",
		},
		{
			name:     "conflicting dispute resolution emits warning",
			ruleCode: "B2B052",
			run:      ConflictingDisputeResolutionRule{}.Run,
			doc: &model.Document{
				Title:      "MSA",
				SourcePath: "msa.docx",
				Text:       "Disputes shall be resolved by arbitration in Mumbai. The courts at Bengaluru shall have exclusive jurisdiction.",
			},
			wantCode: "B2B052",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.run(test.doc, analyze.BuildIndexes(test.doc))
			if len(result) == 0 {
				t.Fatalf("expected diagnostics for rule %s", test.ruleCode)
			}
			if result[0].Code != test.wantCode {
				t.Fatalf("expected %s, got %s", test.wantCode, result[0].Code)
			}
		})
	}
}

func TestConflictingDisputeFixture(t *testing.T) {
	doc, err := docx.Read("../../../testdata/contracts/conflicting-dispute.docx")
	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}

	result := ConflictingDisputeResolutionRule{}.Run(doc, analyze.BuildIndexes(doc))
	if len(result) == 0 {
		t.Fatal("expected conflicting dispute resolution diagnostic")
	}
	if result[0].Code != "B2B052" {
		t.Fatalf("expected B2B052, got %s", result[0].Code)
	}
}

func TestBrokenCrossReferenceFixture(t *testing.T) {
	doc, err := docx.Read("../../../testdata/contracts/broken-cross-reference.docx")
	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}

	result := UnresolvedCrossReferenceRule{}.Run(doc, analyze.BuildIndexes(doc))
	if len(result) == 0 {
		t.Fatal("expected unresolved cross-reference diagnostic")
	}
	if result[0].Code != "STR001" {
		t.Fatalf("expected STR001, got %s", result[0].Code)
	}
}

func TestMissingGoverningLawFixture(t *testing.T) {
	doc, err := docx.Read("../../../testdata/contracts/missing-governing-law.docx")
	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}

	result := MissingGoverningLawOrJurisdictionRule{}.Run(doc, analyze.BuildIndexes(doc))
	if len(result) == 0 {
		t.Fatal("expected missing governing law diagnostic")
	}
	if result[0].Code != "B2B041" {
		t.Fatalf("expected B2B041, got %s", result[0].Code)
	}
}

func TestAmbiguousDaysFixture(t *testing.T) {
	doc, err := docx.Read("../../../testdata/contracts/ambiguous-days.docx")
	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}

	result := AmbiguousTimeComputationRule{}.Run(doc, analyze.BuildIndexes(doc))
	if len(result) == 0 {
		t.Fatal("expected ambiguous days diagnostic")
	}
	if result[0].Code != "LEX012" {
		t.Fatalf("expected LEX012, got %s", result[0].Code)
	}
}
