package main

import (
	// "bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"

	// "io"
	"log"
	"net/http"
	"os"
	pb "request-processor/api/proto"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/grpc"
	"gopkg.in/yaml.v2"
)

type TerraformError struct {
	Message         string // Full error message
	TerraformOutput string // Complete Terraform output including plan/apply details
	Resource        string // Affected resource
}

type RetryConfig struct {
	MaxAttempts int
	Delay       time.Duration
}

type Config struct {
	AnthropicAPIKey string `yaml:"anthropic_api_key"`
	GRPCServerAddr  string `yaml:"grpc_server_addr"`
	Server          struct {
		Port int `yaml:"port"`
	} `yaml:"server"`
}

type TerraformRequest struct {
	Description string `json:"description"`
	Context     string `json:"context"`
	Workspace   string `json:"workspace"`
	Action      string `json:"action"` // "plan", "apply", or "destroy"
}

type TerraformResponse struct {
	Success bool   `json:"success"`
	Code    string `json:"code,omitempty"`
	Output  string `json:"output"`
	Error   string `json:"error,omitempty"`
}

type Service struct {
	anthropicClient *anthropic.Client
	executorClient  pb.ExecutorClient
	config          Config
}

func generateModificationPrompt(description string, existingCode string) string {
	return fmt.Sprintf(`You are a DevOps engineer. There is existing infrastructure that needs modification.
    
	Current Infrastructure:
	%s

	Requested Changes:
	%s

	Requirements:
	1. Only modify the resources mentioned in the request
	2. Keep all other resources unchanged
	3. Preserve resource names and references
	4. Use existing naming conventions
	...`,
		existingCode,
		description,
	)
}

func generateErrorPrompt(originalDescription string, code string, tfError *TerraformError) string {
	return fmt.Sprintf(`You are a DevOps engineer. Previous Terraform code generated an error. Please fix and regenerate the code.

	Original Task: %s

	Previous Code:
	%s

	Terraform Execution Output:
	%s

	Error:
	%s

	Requirements:
	1. Analyze the Terraform execution output and error message
	2. Fix the issues identified in the error messages
	3. Generate ONLY resource and output blocks
	4. DO NOT include:
	- provider configurations
	- terraform blocks
	- variable declarations
	- locals
	5. DO NOT include any explanations or comments
	6. DO NOT include code block markers

	Output ONLY the corrected Terraform code.`,
		originalDescription,
		code,
		tfError.TerraformOutput,
		tfError.Message,
	)
}

func generateInitialInfrastructurePrompt(description string) string {
	return fmt.Sprintf(`You are a DevOps engineer specialized in writing Terraform code. You will receive an infrastructure-related task and must output ONLY the Terraform resource and output blocks - nothing else.

	Task description:
	%s

	Requirements:
	1. If the task is NOT related to infrastructure provisioning, return an empty response
	2. If the task IS infrastructure-related:
	- Generate ONLY resource and output blocks
	- DO NOT include:
		* provider configurations
		* terraform blocks
		* variable declarations
		* locals
		* data sources (unless specifically required)
	- DO NOT include any explanations or comments
	- DO NOT include code block markers (terraform)

	Example task: "Create a droplet in Frankfurt region with 1GB RAM"
	Example output:
	resource "digitalocean_droplet" "web" {
	name   = "web-1"
	region = "fra1"
	size   = "s-1vcpu-1gb"
	image  = "ubuntu-20-04-x64"
	}

	output "droplet_ip" {
	value = digitalocean_droplet.web.ipv4_address
	}

	Your response should contain ONLY Terraform code, nothing else.`, description)
}

func NewService(config Config) (*Service, error) {
	anthropicClient := anthropic.NewClient(
		option.WithAPIKey(config.AnthropicAPIKey),
	)

	conn, err := grpc.Dial(config.GRPCServerAddr, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC server: %v", err)
	}

	executorClient := pb.NewExecutorClient(conn)

	return &Service{
		anthropicClient: anthropicClient,
		executorClient:  executorClient,
		config:          config,
	}, nil
}

func (s *Service) ensureContextAndWorkspace(ctx context.Context, contextName, workspace string) error {
	// Create context if it doesn't exist
	_, err := s.executorClient.CreateContext(ctx, &pb.CreateContextRequest{
		Context: contextName,
	})
	if err != nil {
		return fmt.Errorf("failed to create context: %v", err)
	}

	// Create workspace if it doesn't exist
	_, err = s.executorClient.CreateWorkspace(ctx, &pb.CreateWorkspaceRequest{
		Context:   contextName,
		Workspace: workspace,
	})
	if err != nil {
		return fmt.Errorf("failed to create workspace: %v", err)
	}

	return nil
}

func (s *Service) generateTerraformCode(ctx context.Context, description string, previousError *TerraformError, existingCode string) (string, error) {
	var prompt string
	if previousError != nil {
		prompt = generateErrorPrompt(description, existingCode, previousError)
	} else if existingCode != "" {
		prompt = generateModificationPrompt(description, existingCode)
	} else {
		prompt = generateInitialInfrastructurePrompt(description)
	}

	log.Printf("\n=== LLM Request ===\nDescription: %s\nPrompt:\n%s\n", description, prompt)

	message, err := s.anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.F(anthropic.ModelClaude3_5SonnetLatest),
		MaxTokens: anthropic.F(int64(2048)),
		Messages: anthropic.F([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		}),
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate code: %v", err)
	}

	var code string
	for _, content := range message.Content {
		code += content.Text
	}

	code = strings.TrimPrefix(code, "```hcl")
	code = strings.TrimPrefix(code, "```terraform")
	code = strings.TrimSuffix(code, "```")
	code = strings.TrimSpace(code)

	return code, nil
}

func (s *Service) executeTerraformAction(ctx context.Context, action, description, code, contextName, workspace string) (*TerraformResponse, error) {
	logger := log.New(os.Stdout, "", log.LstdFlags)
	logSection := func(title string) {
		logger.Printf("\n%s %s %s\n", strings.Repeat("=", 10), title, strings.Repeat("=", 10))
	}

	retryConfig := RetryConfig{
		MaxAttempts: 5,
		Delay:       time.Second * 3,
	}

	logSection("Initial Configuration")
	logger.Printf("Action: %s\nContext: %s\nWorkspace: %s", action, contextName, workspace)
	logger.Printf("Initial Code:\n%s", code)

	var lastError error
	lastCode := code
	var response *TerraformResponse

	for attempt := 0; attempt < retryConfig.MaxAttempts; attempt++ {
		logSection(fmt.Sprintf("Attempt %d/%d", attempt+1, retryConfig.MaxAttempts))

		logSection("Workspace Preparation")
		if err := s.prepareWorkspace(ctx, contextName, workspace, lastCode); err != nil {
			logger.Printf("❌ Workspace preparation failed: %v", err)
			return nil, err
		}

		if attempt > 0 && response != nil {
			logSection("Previous Attempt Analysis")
			logger.Printf("Output:\n%s", response.Output)
			logger.Printf("Error:\n%s", response.Error)

			tfError := s.parseTerraformError(response)
			logger.Printf("Parsed Error:\nResource: %s", tfError.Resource)

			newCode, err := s.generateTerraformCode(ctx, description, tfError, lastCode)
			if err != nil {
				logger.Printf("❌ Code generation failed: %v", err)
				lastError = err
				s.logRetryDelay(logger, retryConfig.Delay)
				time.Sleep(retryConfig.Delay)
				continue
			}

			logSection("Code Changes")
			if newCode != lastCode {
				logger.Printf("Changes detected:\nOld:\n%s\n\nNew:\n%s", lastCode, newCode)
			} else {
				logger.Printf("⚠️ Generated code is identical")
			}
			lastCode = newCode
		}

		logSection(fmt.Sprintf("Executing %s", action))
		var err error
		response, err = s.executeAction(ctx, action, contextName, workspace)
		if err != nil {
			logger.Printf("❌ Execution failed: %v", err)
			lastError = err
			s.logRetryDelay(logger, retryConfig.Delay)
			time.Sleep(retryConfig.Delay)
			continue
		}

		if response.Success && response.Error == "" {
			logger.Printf("✅ Action successful!")
			response.Code = lastCode
			return response, nil
		}

		logger.Printf("❌ Attempt failed (Success=%v, Error=%s)", response.Success, response.Error)

		if attempt == retryConfig.MaxAttempts-1 {
			logger.Printf("⚠️ All retry attempts exhausted")
			return response, nil
		}

		s.logRetryDelay(logger, retryConfig.Delay)
		time.Sleep(retryConfig.Delay)
	}

	return response, lastError
}

func (s *Service) prepareWorkspace(ctx context.Context, contextName, workspace, code string) error {
	if _, err := s.executorClient.ClearCode(ctx, &pb.ClearCodeRequest{
		Context:   contextName,
		Workspace: workspace,
	}); err != nil {
		return fmt.Errorf("clear code failed: %v", err)
	}

	if err := s.ensureContextAndWorkspace(ctx, contextName, workspace); err != nil {
		return fmt.Errorf("workspace initialization failed: %v", err)
	}

	if _, err := s.executorClient.AppendCode(ctx, &pb.AppendCodeRequest{
		Context:   contextName,
		Workspace: workspace,
		Code:      code,
	}); err != nil {
		return fmt.Errorf("append code failed: %v", err)
	}

	return nil
}

func (s *Service) parseTerraformError(response *TerraformResponse) *TerraformError {
	tfError := &TerraformError{
		Message:         response.Error,
		TerraformOutput: response.Output,
	}

	if strings.Contains(response.Error, "with") {
		lines := strings.Split(response.Error, "\n")
		for i, line := range lines {
			if strings.Contains(line, "with") && i+2 < len(lines) {
				tfError.Resource = strings.TrimSpace(lines[i+2])
				break
			}
		}
	}
	return tfError
}

func (s *Service) logRetryDelay(logger *log.Logger, delay time.Duration) {
	logger.Printf("⏳ Waiting %v before next attempt...", delay)
}

func (s *Service) executeAction(ctx context.Context, action, contextName, workspace string) (response *TerraformResponse, err error) {
	switch action {
	case "plan":
		resp, err := s.executorClient.Plan(ctx, &pb.PlanRequest{
			Context:   contextName,
			Workspace: workspace,
		})
		if err != nil {
			return nil, err
		}
		return &TerraformResponse{
			Success: resp.Success,
			Output:  resp.PlanOutput,
			Error:   resp.Error,
		}, nil
	case "apply":
		resp, err := s.executorClient.Apply(ctx, &pb.ApplyRequest{
			Context:   contextName,
			Workspace: workspace,
		})
		if err != nil {
			return nil, err
		}
		return &TerraformResponse{
			Success: resp.Success,
			Output:  resp.ApplyOutput,
			Error:   resp.Error,
		}, nil
	case "destroy":
		resp, err := s.executorClient.Destroy(ctx, &pb.DestroyRequest{
			Context:   contextName,
			Workspace: workspace,
		})
		if err != nil {
			return nil, err
		}
		return &TerraformResponse{
			Success: resp.Success,
			Output:  resp.DestroyOutput,
			Error:   resp.Error,
		}, nil

	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

func (s *Service) handleTerraformRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req TerraformRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Context == "" {
		req.Context = "default"
	}
	if req.Action == "" {
		req.Action = "plan"
	}

	var code string
	var err error

	if req.Action != "destroy" {
		existingCode, err := s.executorClient.GetMainTf(r.Context(), &pb.GetMainTfRequest{
			Context:   req.Context,
			Workspace: req.Workspace,
		})

		codeContent := ""
		if err == nil { // Если код существует
			codeContent = existingCode.Content
		}
		if !(req.Action == "apply" && req.Description == "") {
			code, err = s.generateTerraformCode(r.Context(), req.Description, nil, codeContent)
			if err != nil {
				http.Error(w, fmt.Sprintf("Failed to generate code: %v", err), http.StatusInternalServerError)
				return
			}
		} else {
			req.Description = "Please check that code is correct"
			code = codeContent
		}

	}

	response, err := s.executeTerraformAction(r.Context(), req.Action, req.Description, code, req.Context, req.Workspace)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to execute terraform action: %v", err), http.StatusInternalServerError)
		return
	}

	if code != "" {
		response.Code = code
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func LoadConfig(filename string) (*Config, error) {
	config := &Config{}

	buf, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %v", err)
	}

	err = yaml.Unmarshal(buf, config)
	if err != nil {
		return nil, fmt.Errorf("error parsing config file: %v", err)
	}

	if env := os.Getenv("ANTHROPIC_API_KEY"); env != "" {
		config.AnthropicAPIKey = env
	}
	if env := os.Getenv("GRPC_SERVER_ADDR"); env != "" {
		config.GRPCServerAddr = env
	}

	if config.AnthropicAPIKey == "" {
		return nil, fmt.Errorf("anthropic_api_key is required")
	}
	if config.GRPCServerAddr == "" {
		config.GRPCServerAddr = "localhost:50051"
	}
	if config.Server.Port == 0 {
		config.Server.Port = 8080
	}

	return config, nil
}

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	config, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	service, err := NewService(*config)
	if err != nil {
		log.Fatalf("Failed to create service: %v", err)
	}

	http.HandleFunc("/terraform", service.handleTerraformRequest)
	serverAddr := fmt.Sprintf(":%d", config.Server.Port)
	log.Printf("Server starting on %s", serverAddr)
	if err := http.ListenAndServe(serverAddr, nil); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
