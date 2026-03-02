package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/progress"
)

// HardeningProvenance holds metadata about the hardening pipeline that produced a model.
type HardeningProvenance struct {
	SourceModel     string  `json:"source_model"`
	HardenedModel   string  `json:"hardened_model"`
	ScanFindings    int     `json:"scan_findings,omitempty"`
	FixRate         float64 `json:"fix_rate,omitempty"`
	Verdict         string  `json:"verdict,omitempty"`
	Regressions     int     `json:"regressions,omitempty"`
	PipelineVersion string  `json:"pipeline_version,omitempty"`
	Timestamp       string  `json:"timestamp"`
}

// ImportHfHandler handles the import-hf CLI command.
// It downloads a model from HuggingFace, converts to GGUF, generates a Modelfile,
// and optionally creates and pushes the model.
func ImportHfHandler(cmd *cobra.Command, args []string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	hfModel := args[0]
	quant, _ := cmd.Flags().GetString("quant")
	system, _ := cmd.Flags().GetString("system")
	tag, _ := cmd.Flags().GetString("tag")
	push, _ := cmd.Flags().GetBool("push")
	sourceModel, _ := cmd.Flags().GetString("source-model")
	scanFindings, _ := cmd.Flags().GetInt("scan-findings")
	fixRate, _ := cmd.Flags().GetFloat64("fix-rate")
	verdict, _ := cmd.Flags().GetString("verdict")
	pipelineVersion, _ := cmd.Flags().GetString("pipeline-version")

	if quant == "" {
		quant = "q4_k_m"
	}

	// Derive tag from HF model name if not specified
	if tag == "" {
		parts := strings.Split(hfModel, "/")
		modelName := parts[len(parts)-1]
		modelName = strings.ToLower(modelName)
		modelName = strings.ReplaceAll(modelName, " ", "-")
		tag = "hardened/" + modelName + ":latest"
	}

	// Create working directory
	workDir, err := os.MkdirTemp("", "armyknife-import-*")
	if err != nil {
		return fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Step 1: Download from HuggingFace
	spinner := progress.NewSpinner("Downloading from HuggingFace")
	p.Add("download", spinner)

	downloadDir := filepath.Join(workDir, "hf-model")
	if err := downloadHfModel(hfModel, downloadDir); err != nil {
		spinner.SetMessage("Download failed")
		spinner.Stop()
		return fmt.Errorf("failed to download from HuggingFace: %w", err)
	}
	spinner.SetMessage("Download complete")
	spinner.Stop()

	// Step 2: Convert to GGUF
	spinner = progress.NewSpinner("Converting to GGUF (f16)")
	p.Add("convert", spinner)

	ggufF16 := filepath.Join(workDir, "model-f16.gguf")
	if err := convertToGGUF(downloadDir, ggufF16); err != nil {
		spinner.SetMessage("Conversion failed")
		spinner.Stop()
		return fmt.Errorf("failed to convert to GGUF: %w", err)
	}
	spinner.SetMessage("Conversion complete")
	spinner.Stop()

	// Step 3: Quantize
	ggufQuant := filepath.Join(workDir, fmt.Sprintf("model-%s.gguf", quant))
	spinner = progress.NewSpinner(fmt.Sprintf("Quantizing to %s", quant))
	p.Add("quantize", spinner)

	if err := quantizeGGUF(ggufF16, ggufQuant, quant); err != nil {
		spinner.SetMessage("Quantization failed")
		spinner.Stop()
		return fmt.Errorf("failed to quantize: %w", err)
	}
	spinner.SetMessage("Quantization complete")
	spinner.Stop()

	// Step 4: Generate Modelfile
	provenance := &HardeningProvenance{
		SourceModel:     sourceModel,
		HardenedModel:   hfModel,
		ScanFindings:    scanFindings,
		FixRate:         fixRate,
		Verdict:         verdict,
		PipelineVersion: pipelineVersion,
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
	}

	modelfilePath := filepath.Join(workDir, "Modelfile")
	if err := generateModelfile(ggufQuant, system, provenance, modelfilePath); err != nil {
		return fmt.Errorf("failed to generate Modelfile: %w", err)
	}

	// Step 5: Create model via ollama create
	spinner = progress.NewSpinner(fmt.Sprintf("Creating model %s", tag))
	p.Add("create", spinner)

	if err := createModel(tag, modelfilePath); err != nil {
		spinner.SetMessage("Create failed")
		spinner.Stop()
		return fmt.Errorf("failed to create model: %w", err)
	}
	spinner.SetMessage("Model created")
	spinner.Stop()

	// Step 6: Optionally push
	if push {
		reg := envconfig.ArmyknifeRegistry()
		if reg == "" {
			reg = "default registry"
		}
		spinner = progress.NewSpinner(fmt.Sprintf("Pushing to %s", reg))
		p.Add("push", spinner)

		if err := pushModel(tag); err != nil {
			spinner.SetMessage("Push failed")
			spinner.Stop()
			return fmt.Errorf("failed to push model: %w", err)
		}
		spinner.SetMessage("Push complete")
		spinner.Stop()
	}

	p.Stop()

	fmt.Fprintf(os.Stderr, "\nImport complete: %s\n", tag)
	fmt.Fprintf(os.Stderr, "Run with: armyknife-ollama run %s\n", tag)

	return nil
}

// downloadHfModel downloads a model from HuggingFace Hub.
func downloadHfModel(modelID, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	// Try huggingface-cli first
	if path, err := exec.LookPath("huggingface-cli"); err == nil {
		cmd := exec.Command(path, "download", modelID, "--local-dir", destDir)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}

	// Fall back to git clone with LFS
	if _, err := exec.LookPath("git"); err == nil {
		url := fmt.Sprintf("https://huggingface.co/%s", modelID)
		cmd := exec.Command("git", "clone", "--depth=1", url, destDir)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		cmd.Env = append(os.Environ(), "GIT_LFS_SKIP_SMUDGE=0")
		return cmd.Run()
	}

	return fmt.Errorf("neither huggingface-cli nor git found; install one to download models")
}

// convertToGGUF converts a HuggingFace model to GGUF format using convert_hf_to_gguf.py.
func convertToGGUF(modelDir, outputPath string) error {
	// Look for convert_hf_to_gguf.py in common locations
	candidates := []string{
		"convert_hf_to_gguf.py",
		filepath.Join(os.Getenv("HOME"), "llama.cpp", "convert_hf_to_gguf.py"),
		"/usr/local/bin/convert_hf_to_gguf.py",
	}

	var scriptPath string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			scriptPath = c
			break
		}
		if p, err := exec.LookPath(c); err == nil {
			scriptPath = p
			break
		}
	}

	if scriptPath == "" {
		return fmt.Errorf("convert_hf_to_gguf.py not found; install llama.cpp or set it in PATH")
	}

	python := "python3"
	if p, err := exec.LookPath("python3"); err != nil {
		if p, err = exec.LookPath("python"); err != nil {
			return fmt.Errorf("python3 not found")
		}
		python = p
	} else {
		python = p
	}

	cmd := exec.Command(python, scriptPath, modelDir, "--outtype", "f16", "--outfile", outputPath)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// quantizeGGUF quantizes a GGUF model using llama-quantize.
func quantizeGGUF(inputPath, outputPath, quantType string) error {
	quantize := "llama-quantize"
	if p, err := exec.LookPath("llama-quantize"); err != nil {
		// Try llama.cpp build directory
		home := os.Getenv("HOME")
		alt := filepath.Join(home, "llama.cpp", "build", "bin", "llama-quantize")
		if _, err := os.Stat(alt); err == nil {
			quantize = alt
		} else {
			return fmt.Errorf("llama-quantize not found; install llama.cpp or set it in PATH")
		}
	} else {
		quantize = p
	}

	cmd := exec.Command(quantize, inputPath, outputPath, strings.ToUpper(quantType))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// generateModelfile creates a Modelfile for the converted model.
func generateModelfile(ggufPath, system string, provenance *HardeningProvenance, outputPath string) error {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("FROM %s\n\n", ggufPath))

	if system != "" {
		sb.WriteString(fmt.Sprintf("SYSTEM \"\"\"%s\"\"\"\n\n", system))
	}

	// Default parameters for hardened models
	sb.WriteString("PARAMETER temperature 0.7\n")
	sb.WriteString("PARAMETER top_p 0.9\n")
	sb.WriteString("PARAMETER stop \"<|im_end|>\"\n")
	sb.WriteString("PARAMETER stop \"<|endoftext|>\"\n")
	sb.WriteString("\n")

	// Add provenance as a license/metadata block
	if provenance != nil && provenance.HardenedModel != "" {
		provenanceJSON, err := json.MarshalIndent(provenance, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal provenance: %w", err)
		}
		sb.WriteString(fmt.Sprintf("LICENSE \"\"\"\nArmyKnife Hardening Provenance:\n%s\n\"\"\"\n", string(provenanceJSON)))
	}

	return os.WriteFile(outputPath, []byte(sb.String()), 0o644)
}

// createModel shells out to the current binary to create a model from a Modelfile.
func createModel(tag, modelfilePath string) error {
	self, err := os.Executable()
	if err != nil {
		self = "armyknife-ollama"
	}

	cmd := exec.Command(self, "create", tag, "-f", modelfilePath)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// pushModel shells out to the current binary to push a model to the registry.
func pushModel(tag string) error {
	self, err := os.Executable()
	if err != nil {
		self = "armyknife-ollama"
	}

	cmd := exec.Command(self, "push", tag)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
