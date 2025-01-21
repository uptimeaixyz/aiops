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

type Config struct {
	AnthropicAPIKey string `yaml:"anthropic_api_key"`
	GRPCServerAddr  string `yaml:"grpc_server_addr"`
	Server          struct {
		Port int `yaml:"port"`
	} `yaml:"server"`
}

type Service struct {
	anthropicClient *anthropic.Client
	executorClient  pb.ExecutorClient
	config          Config
}

type TerraformRequest struct {
	Description string            `json:"description"`
	Context     string           `json:"context"`
	Workspace   string            `json:"workspace"`
	Action      string           `json:"action"` // "plan", "apply", or "destroy"
	Variables   map[string]string `json:"variables,omitempty"`
}

type TerraformResponse struct {
	Success bool   `json:"success"`
	Code    string `json:"code,omitempty"`
	Output  string `json:"output"`
	Error   string `json:"error,omitempty"`
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

// func (s *Service) configureProviders(ctx context.Context, contextName, workspace string) error {
// 	// Configure DigitalOcean provider
// 	_, err := s.executorClient.AddProviders(ctx, &pb.AddProvidersRequest{
// 		Context:   contextName,
// 		Workspace: workspace,
// 		Providers: []*pb.Provider{
// 			{
// 				Name:    "digitalocean",
// 				Source:  "digitalocean/digitalocean",
// 				Version: "~> 2.0",
// 			},
// 		},
// 	})
// 	return err
// }

func (s *Service) generateTerraformCode(ctx context.Context, description string) (string, error) {
	prompt := fmt.Sprintf(`You are a DevOps engineer specialized in writing Terraform code. You will receive an infrastructure-related task and must output ONLY the Terraform resource and output blocks - nothing else.

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
   - DO NOT include code block markers (``terraform or ``)

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

	code = strings.TrimPrefix(code, "```hcl")
	code = strings.TrimPrefix(code, "```terraform")
	code = strings.TrimSuffix(code, "```")
	code = strings.TrimSpace(code)

	return code, nil
}

func (s *Service) executeTerraformAction(ctx context.Context, action, code, contextName, workspace string) (*TerraformResponse, error) {
	// First ensure context and workspace exist
	if err := s.ensureContextAndWorkspace(ctx, contextName, workspace); err != nil {
		return nil, err
	}

	// Clear any existing code
	_, err := s.executorClient.ClearCode(ctx, &pb.ClearCodeRequest{
		Context:   contextName,
		Workspace: workspace,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to clear existing code: %v", err)
	}

	// // Configure providers
	// if err := s.configureProviders(ctx, contextName, workspace); err != nil {
	// 	return nil, fmt.Errorf("failed to configure providers: %v", err)
	// }

	// Append new code if provided
	if code != "" {
		_, err = s.executorClient.AppendCode(ctx, &pb.AppendCodeRequest{
			Context:   contextName,
			Workspace: workspace,
			Code:      code,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to append code: %v", err)
		}
	}

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
		code, err = s.generateTerraformCode(r.Context(), req.Description)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate code: %v", err), http.StatusInternalServerError)
			return
		}
	}

	response, err := s.executeTerraformAction(r.Context(), req.Action, code, req.Context, req.Workspace)
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
curl -X POST http://localhost:8080/terraform \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Create digitalocean droplet with 1GB RAM in Frankfurt",
    "context": "segovchik",
	"workspace": "airdao",
  }'