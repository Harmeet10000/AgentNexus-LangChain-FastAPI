package ports

import "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"

type Rule interface {
	Code() string
	Run(doc *model.Document, idx *model.Indexes) []model.Diagnostic
}
