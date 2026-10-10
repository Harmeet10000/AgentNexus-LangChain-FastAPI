package main

import (
	"fmt"
	"os"

	"github.com/harmeet/langchain-fastapi-production/lynk/internal/adapters/cli"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	switch os.Args[1] {
	case "lint":
		os.Exit(cli.RunLint(os.Args[2:]))
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: lynk <command>")
	fmt.Fprintln(os.Stderr, "commands:")
	fmt.Fprintln(os.Stderr, "  lint    lint a .docx contract")
}
