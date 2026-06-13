package model

type Severity string

const (
	SeverityInfo    Severity = "info"
	SeverityWarning Severity = "warning"
	SeverityError   Severity = "error"
)

type Suggestion struct {
	Summary     string `json:"summary"`
	Replacement string `json:"replacement,omitempty"`
	Pattern     string `json:"pattern,omitempty"`
}

type Diagnostic struct {
	Code       string      `json:"code"`
	Severity   Severity    `json:"severity"`
	Message    string      `json:"message"`
	Evidence   string      `json:"evidence,omitempty"`
	Section    string      `json:"section,omitempty"`
	Paragraph  int         `json:"paragraph,omitempty"`
	Suggestion *Suggestion `json:"suggestion,omitempty"`
}

type LintResult struct {
	DocumentTitle string       `json:"document_title"`
	DocumentPath  string       `json:"document_path"`
	Diagnostics   []Diagnostic `json:"diagnostics"`
}
