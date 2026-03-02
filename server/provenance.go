package server

import (
	"encoding/json"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// ProvenanceResponse is the API response for /api/provenance/:model.
type ProvenanceResponse struct {
	Model      string          `json:"model"`
	Provenance json.RawMessage `json:"provenance,omitempty"`
	Error      string          `json:"error,omitempty"`
}

// ProvenanceHandler returns hardening provenance metadata for a model.
func (s *Server) ProvenanceHandler(c *gin.Context) {
	modelName := c.Param("model")
	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model name required"})
		return
	}

	name := model.ParseName(modelName)
	if !name.IsValid() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid model name"})
		return
	}

	m, err := manifest.ParseNamedManifest(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "model not found"})
		return
	}

	// Search layers for provenance metadata
	for _, layer := range m.Layers {
		if layer.MediaType == manifest.MediaTypeHardeningProvenance {
			blobPath, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to resolve provenance blob"})
				return
			}

			data, err := os.ReadFile(blobPath)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read provenance data"})
				return
			}

			c.JSON(http.StatusOK, ProvenanceResponse{
				Model:      modelName,
				Provenance: json.RawMessage(data),
			})
			return
		}
	}

	c.JSON(http.StatusNotFound, ProvenanceResponse{
		Model: modelName,
		Error: "no hardening provenance metadata found for this model",
	})
}
