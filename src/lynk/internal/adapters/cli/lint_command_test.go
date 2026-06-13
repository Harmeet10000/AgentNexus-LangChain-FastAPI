package cli

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestLintJSONGolden(t *testing.T) {
	fixturePath := filepath.Join("..", "..", "..", "testdata", "contracts", "conflicting-dispute.docx")
	goldenPath := filepath.Join("..", "..", "..", "testdata", "golden", "conflicting-dispute.json")
	command := exec.Command(
		"go",
		"run",
		"../../../cmd/lynk",
		"lint",
		"--format",
		"json",
		fixturePath,
	)
	command.Dir = "."
	output, err := command.Output()
	if err != nil {
		t.Fatalf("expected CLI JSON output, got error: %v", err)
	}

	golden, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("expected golden file, got error: %v", err)
	}
	golden = bytes.ReplaceAll(golden, []byte("__FIXTURE_PATH__"), []byte(fixturePath))

	if !bytes.Equal(bytes.TrimSpace(output), bytes.TrimSpace(golden)) {
		t.Fatalf("json output mismatch\nwant:\n%s\n\ngot:\n%s", golden, output)
	}
}
