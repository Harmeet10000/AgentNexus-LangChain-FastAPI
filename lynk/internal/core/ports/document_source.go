package ports

import "github.com/harmeet/langchain-fastapi-production/lynk/internal/core/model"

type DocumentSource interface {
	Read(path string) (*model.Document, error)
}
