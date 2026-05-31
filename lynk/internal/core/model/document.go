package model

type Span struct {
	Start int
	End   int
}

type Paragraph struct {
	Index      int
	Text       string
	Style      string
	HeadingLvl int
	Span       Span
}

type Section struct {
	Index      int
	Title      string
	Number     string
	Level      int
	Paragraphs []int
	Span       Span
}

type Document struct {
	SourcePath string
	Title      string
	Text       string
	Paragraphs []Paragraph
	Sections   []Section
	Metadata   map[string]string
	Hashes     map[string]string
}
