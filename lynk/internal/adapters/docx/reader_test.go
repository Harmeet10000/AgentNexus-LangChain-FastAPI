package docx

import "testing"

func TestNormalizeWhitespace(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "mixed whitespace collapses",
			input: "  Governing   law\n\t clause  ",
			want:  "Governing law clause",
		},
		{
			name:  "empty stays empty",
			input: "   \n\t   ",
			want:  "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := normalizeWhitespace(test.input)
			if got != test.want {
				t.Fatalf("normalizeWhitespace(%q) = %q, want %q", test.input, got, test.want)
			}
		})
	}
}

func TestSplitSectionHeading(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		wantNumber string
		wantTitle  string
	}{
		{
			name:       "numbered heading",
			input:      "9. Governing Law and Jurisdiction",
			wantNumber: "9",
			wantTitle:  "Governing Law and Jurisdiction",
		},
		{
			name:       "plain heading",
			input:      "Confidentiality",
			wantNumber: "",
			wantTitle:  "Confidentiality",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			number, title := splitSectionHeading(test.input)
			if number != test.wantNumber {
				t.Fatalf("number = %q, want %q", number, test.wantNumber)
			}
			if title != test.wantTitle {
				t.Fatalf("title = %q, want %q", title, test.wantTitle)
			}
		})
	}
}

func TestReadFixtureDocument(t *testing.T) {
 	doc, err := Read("../../../testdata/contracts/minimal-valid.docx")
 	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}
	if doc.Title == "" {
		t.Fatal("expected fixture title")
	}
	if len(doc.Paragraphs) == 0 {
		t.Fatal("expected parsed paragraphs")
	}
}

func TestReadAdditionalFixtures(t *testing.T) {
	fixtures := []string{
		"../../../testdata/contracts/broken-cross-reference.docx",
		"../../../testdata/contracts/missing-governing-law.docx",
		"../../../testdata/contracts/ambiguous-days.docx",
	}

	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			doc, err := Read(fixture)
			if err != nil {
				t.Fatalf("expected fixture to parse, got error: %v", err)
			}
			if doc.Title == "" {
				t.Fatal("expected fixture title")
			}
			if len(doc.Paragraphs) == 0 {
				t.Fatal("expected parsed paragraphs")
			}
		})
	}
}
