package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	pb "request-processor/api/proto"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/grpc"
	"gopkg.in/yaml.v2"
)

// Configuration struct
type Config struct {
	AnthropicAPIKey string `yaml:"anthropic_api_key"`
	GRPCServerAddr  string `yaml:"grpc_server_addr"`
	Server          struct {
		Port int `yaml:"port"`
	} `yaml:"server"`
}

// Service struct to hold dependencies
type Service struct {
	anthropicClient *anthropic.Client
	executorClient  pb.ExecutorClient
	config          Config
}

// Request structure for the HTTP API
type TerraformRequest struct {
	Description string            `json:"description"`
	Workspace   string            `json:"workspace"`
	Action      string            `json:"action"` // "plan", "apply", or "destroy"
	Variables   map[string]string `json:"variables,omitempty"`
}

// Response structure for the HTTP API
type TerraformResponse struct {
	Success bool   `json:"success"`
	Code    string `json:"code,omitempty"`
	Output  string `json:"output"`
	Error   string `json:"error,omitempty"`
}

// NewService creates a new instance of the service
func NewService(config Config) (*Service, error) {
	// Initialize Anthropic client
	anthropicClient := anthropic.NewClient(
		option.WithAPIKey(config.AnthropicAPIKey),
	)

	// Set up gRPC connection
	conn, err := grpc.Dial(config.GRPCServerAddr, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC server: %v", err)
	}

	// Create executor client
	executorClient := pb.NewExecutorClient(conn)

	return &Service{
		anthropicClient: anthropicClient,
		executorClient:  executorClient,
		config:          config,
	}, nil
}

func (s *Service) generateTerraformCode(ctx context.Context, description string) (string, error) {
	prompt := fmt.Sprintf(`You are a DevOps engineer. OUTPUT ONLY TERRAFORM CODE, NO OTHER EXPLANATIONS.

Create Terraform code for DigitalOcean based on this description:
%s
The task:
- Create digitalocean droplet, frankfurt, 1gb ram, 1 cpu
Requirements:
Generate Terraform resource and output blocks ONLY for DigitalOcean infrastructure. DO NOT include any terraform blocks, provider configurations, or variable declarations - they are already set up. The environment already has:

DigitalOcean provider configured
Backend configuration
do_token variable
OUTPUT ONLY the resource and output blocks that define the infrastructure."

Then after this prompt, you would add the specific infrastructure requirements like "Create a droplet in Frankfurt" or "Create a Kubernetes cluster with 3 nodes

OUTPUT ONLY TERRAFORM CODE, NO OTHER EXPLANATIONS`, description)

	message, err := s.anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.F(anthropic.ModelClaude3_5SonnetLatest),
		MaxTokens: anthropic.F(int64(1024)),
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

	// Clean up the code
	code = strings.TrimPrefix(code, "```hcl")
	code = strings.TrimPrefix(code, "```terraform")
	code = strings.TrimSuffix(code, "```")
	code = strings.TrimSpace(code)

	// Don't escape newlines or quotes - they should be preserved as-is for Terraform
	return code, nil
}

func (s *Service) executeTerraformAction(ctx context.Context, action string, code string, workspace string) (*TerraformResponse, error) {
	switch action {
	case "plan":
		resp, err := s.executorClient.Plan(ctx, &pb.PlanRequest{
			Workspace: workspace,
			Code:      code,
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
			Workspace: workspace,
			Code:      code,
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

	// Set default action to "plan" if not specified
	if req.Action == "" {
		req.Action = "plan"
	}

	var code string
	var err error

	// Only generate code for plan and apply actions
	if req.Action != "destroy" {
		code, err = s.generateTerraformCode(r.Context(), req.Description)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate code: %v", err), http.StatusInternalServerError)
			return
		}
	}

	// Execute the requested action
	response, err := s.executeTerraformAction(r.Context(), req.Action, code, req.Workspace)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to execute terraform action: %v", err), http.StatusInternalServerError)
		return
	}

	// Add the generated code to the response if it exists
	if code != "" {
		response.Code = code
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// LoadConfig loads configuration from a YAML file
func LoadConfig(filename string) (*Config, error) {
	// Check environment variables first
	config := &Config{}

	// Read from file
	buf, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %v", err)
	}

	err = yaml.Unmarshal(buf, config)
	if err != nil {
		return nil, fmt.Errorf("error parsing config file: %v", err)
	}

	// Override with environment variables if they exist
	if env := os.Getenv("ANTHROPIC_API_KEY"); env != "" {
		config.AnthropicAPIKey = env
	}
	if env := os.Getenv("GRPC_SERVER_ADDR"); env != "" {
		config.GRPCServerAddr = env
	}

	// Validate required fields
	if config.AnthropicAPIKey == "" {
		return nil, fmt.Errorf("anthropic_api_key is required")
	}
	if config.GRPCServerAddr == "" {
		config.GRPCServerAddr = "localhost:50051" // default value
	}
	if config.Server.Port == 0 {
		config.Server.Port = 8080 // default value
	}

	return config, nil
}

func main() {
	// Parse command line flags
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	// Load configuration
	config, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	service, err := NewService(Config{
		AnthropicAPIKey: config.AnthropicAPIKey,
		GRPCServerAddr:  config.GRPCServerAddr,
	})
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
