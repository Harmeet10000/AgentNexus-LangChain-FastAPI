package docx

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"
)

var sectionNumberPattern = regexp.MustCompile(`^(\d+(?:\.\d+)*)[.)]?\s+(.+)$`)

const maxDocxSizeBytes = 25 * 1024 * 1024
const maxDocxEntryCount = 128
const maxCompressionRatio = 100.0

type xmlRunProperties struct {
	Bold   *struct{} `xml:"b"`
	Italic *struct{} `xml:"i"`
	Size   string    `xml:"sz,omitempty"`
	Fonts  string    `xml:"rFonts,omitempty"`
	Style  string    `xml:"rStyle,omitempty"`
	Caps   *struct{} `xml:"caps"`
	Small  *struct{} `xml:"smallCaps"`
}

type xmlParagraphProperties struct {
	Style xmlParagraphStyle `xml:"pStyle"`
}

type xmlParagraphStyle struct {
	Val string `xml:"val,attr"`
}

type Reader struct{}

func (Reader) Read(path string) (*model.Document, error) {
	return Read(path)
}

func Read(path string) (*model.Document, error) {
	rawBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read docx bytes: %w", err)
	}
	if len(rawBytes) > maxDocxSizeBytes {
		return nil, fmt.Errorf("docx exceeds maximum supported size of %d bytes", maxDocxSizeBytes)
	}

	reader, err := zip.NewReader(bytes.NewReader(rawBytes), int64(len(rawBytes)))
	if err != nil {
		return nil, fmt.Errorf("open docx zip: %w", err)
	}
	if len(reader.File) > maxDocxEntryCount {
		return nil, fmt.Errorf("docx contains too many archive entries: %d", len(reader.File))
	}

	var documentXML io.ReadCloser
	for _, file := range reader.File {
		if file.CompressedSize64 > 0 {
			ratio := float64(file.UncompressedSize64) / float64(file.CompressedSize64)
			if ratio > maxCompressionRatio {
				return nil, fmt.Errorf("docx entry %s exceeds maximum compression ratio", file.Name)
			}
		}
		if strings.HasPrefix(file.Name, "word/_rels/") || strings.HasPrefix(file.Name, "_rels/") {
			continue
		}
		if file.Name == "word/document.xml" {
			if file.UncompressedSize64 > maxDocxSizeBytes {
				return nil, fmt.Errorf("document.xml exceeds maximum supported size of %d bytes", maxDocxSizeBytes)
			}
			documentXML, err = file.Open()
			if err != nil {
				return nil, fmt.Errorf("open document.xml: %w", err)
			}
			defer documentXML.Close()
			break
		}
	}
	if documentXML == nil {
		return nil, fmt.Errorf("document.xml not found in %s", path)
	}

	decoder := xml.NewDecoder(documentXML)
	doc := &model.Document{
		SourcePath: path,
		Metadata:   map[string]string{"format": "docx"},
		Hashes: map[string]string{
			"sha256": sha256Hex(rawBytes),
		},
	}

	var currentParagraph strings.Builder
	var currentStyle string
	var currentHeadingLevel int
	var paragraphIndex int
	var textOffset int
	var inParagraph bool

	flushParagraph := func() {
		text := normalizeWhitespace(currentParagraph.String())
		if strings.TrimSpace(text) == "" {
			currentParagraph.Reset()
			currentStyle = ""
			currentHeadingLevel = 0
			return
		}

		span := model.Span{Start: textOffset, End: textOffset + len(text)}
		paragraph := model.Paragraph{
			Index:      paragraphIndex,
			Text:       text,
			Style:      currentStyle,
			HeadingLvl: currentHeadingLevel,
			Span:       span,
		}
		doc.Paragraphs = append(doc.Paragraphs, paragraph)
		if doc.Text == "" {
			doc.Text = text
		} else {
			doc.Text += "\n\n" + text
		}
		textOffset = len(doc.Text)
		paragraphIndex++
		currentParagraph.Reset()
		currentStyle = ""
		currentHeadingLevel = 0
	}

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stream xml token: %w", err)
		}

		switch typed := token.(type) {
		case xml.StartElement:
			switch typed.Name.Local {
			case "p":
				inParagraph = true
			case "pPr":
				if !inParagraph {
					continue
				}
				var props xmlParagraphProperties
				if err := decoder.DecodeElement(&props, &typed); err != nil {
					return nil, fmt.Errorf("decode paragraph properties: %w", err)
				}
				currentStyle = props.Style.Val
				currentHeadingLevel = headingLevel(props.Style.Val)
			case "t":
				if !inParagraph {
					continue
				}
				var text string
				if err := decoder.DecodeElement(&text, &typed); err != nil {
					return nil, fmt.Errorf("decode text run: %w", err)
				}
				currentParagraph.WriteString(text)
				currentParagraph.WriteString(" ")
			}
		case xml.EndElement:
			if typed.Name.Local == "p" {
				flushParagraph()
				inParagraph = false
			}
		}
	}

	buildSections(doc)
	if doc.Title == "" {
		doc.Title = inferTitle(doc)
	}

	return doc, nil
}

func buildSections(doc *model.Document) {
	sections := make([]model.Section, 0)
	current := model.Section{Index: -1}

	flush := func(endSpan int) {
		if current.Index < 0 {
			return
		}
		current.Span.End = endSpan
		sections = append(sections, current)
	}

	for _, paragraph := range doc.Paragraphs {
		if !isSectionHeading(paragraph) {
			if current.Index >= 0 {
				current.Paragraphs = append(current.Paragraphs, paragraph.Index)
				current.Span.End = paragraph.Span.End
			}
			continue
		}

		flush(paragraph.Span.Start)
		number, title := splitSectionHeading(paragraph.Text)
		current = model.Section{
			Index:      len(sections),
			Title:      title,
			Number:     number,
			Level:      max(paragraph.HeadingLvl, 1),
			Paragraphs: []int{paragraph.Index},
			Span:       model.Span{Start: paragraph.Span.Start, End: paragraph.Span.End},
		}
	}

	flush(len(doc.Text))
	doc.Sections = sections
}

func inferTitle(doc *model.Document) string {
	for _, paragraph := range doc.Paragraphs {
		if paragraph.HeadingLvl > 0 || paragraph.Index == 0 {
			return paragraph.Text
		}
	}
	return "Untitled Contract"
}

func headingLevel(style string) int {
	style = strings.ToLower(strings.TrimSpace(style))
	switch {
	case strings.HasPrefix(style, "heading1") || style == "title":
		return 1
	case strings.HasPrefix(style, "heading2"):
		return 2
	case strings.HasPrefix(style, "heading3"):
		return 3
	default:
		return 0
	}
}

func isSectionHeading(paragraph model.Paragraph) bool {
	if paragraph.HeadingLvl > 0 {
		return true
	}
	return sectionNumberPattern.MatchString(paragraph.Text)
}

func splitSectionHeading(text string) (string, string) {
	matches := sectionNumberPattern.FindStringSubmatch(text)
	if len(matches) == 3 {
		return matches[1], matches[2]
	}
	return "", text
}

func normalizeWhitespace(text string) string {
	fields := strings.Fields(text)
	return strings.TrimSpace(strings.Join(fields, " "))
}

func sha256Hex(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
