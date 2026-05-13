package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/spikeon/llm-router/internal/config"
	"github.com/spikeon/llm-router/internal/ollama"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/drive/v3"
	"google.golang.org/api/gmail/v1"
	"google.golang.org/api/option"
	"google.golang.org/api/sheets/v4"
)

var googleScopes = []string{
	"https://www.googleapis.com/auth/spreadsheets.readonly",
	"https://www.googleapis.com/auth/drive.readonly",
	"https://www.googleapis.com/auth/gmail.readonly",
}

// GetAPIKey returns the Gemini API key from GEMINI_API_KEY env, or a JSON
// credential pool file at GEMINI_AUTH_JSON (defaults to ~/.hermes/auth.json).
func GetAPIKey() string {
	if k := os.Getenv("GEMINI_API_KEY"); k != "" {
		return k
	}
	authJSON := os.Getenv("GEMINI_AUTH_JSON")
	if authJSON == "" {
		authJSON = os.ExpandEnv("$HOME/.hermes/auth.json")
	}
	data, err := os.ReadFile(authJSON)
	if err != nil {
		return ""
	}
	var auth map[string]any
	if err := json.Unmarshal(data, &auth); err != nil {
		return ""
	}
	pool, _ := auth["credential_pool"].(map[string]any)
	gemini, _ := pool["gemini"].([]any)
	if len(gemini) == 0 {
		return ""
	}
	entry, _ := gemini[0].(map[string]any)
	token, _ := entry["access_token"].(string)
	return token
}

// resolveGooglePaths returns (tokenPath, credentialsPath).
// Priority: GOOGLE_TOKEN_PATH / GOOGLE_CREDENTIALS_PATH env vars →
// ~/.hermes/google_token.json (if present) → ~/.config/spikeon-router fallback.
func resolveGooglePaths() (string, string) {
	if t, s := os.Getenv("GOOGLE_TOKEN_PATH"), os.Getenv("GOOGLE_CREDENTIALS_PATH"); t != "" && s != "" {
		return t, s
	}
	hermesToken := os.ExpandEnv("$HOME/.hermes/google_token.json")
	hermesSecret := os.ExpandEnv("$HOME/.hermes/google_client_secret.json")
	fallbackToken := os.ExpandEnv("$HOME/.config/spikeon-router/google_token.json")
	fallbackSecret := os.ExpandEnv("$HOME/.config/spikeon-router/google_credentials.json")

	tokenPath, secretPath := fallbackToken, fallbackSecret
	if _, err := os.Stat(hermesToken); err == nil {
		tokenPath = hermesToken
	}
	if _, err := os.Stat(hermesSecret); err == nil {
		secretPath = hermesSecret
	}
	return tokenPath, secretPath
}

func tokenSource() (oauth2.TokenSource, error) {
	tokenPath, secretPath := resolveGooglePaths()

	secretData, err := os.ReadFile(secretPath)
	if err != nil {
		return nil, fmt.Errorf("no Google credentials at %s", secretPath)
	}
	cfg, err := google.ConfigFromJSON(secretData, googleScopes...)
	if err != nil {
		return nil, err
	}
	tokenData, err := os.ReadFile(tokenPath)
	if err != nil {
		return nil, fmt.Errorf("no Google token at %s — run OAuth flow first", tokenPath)
	}
	var tok oauth2.Token
	if err := json.Unmarshal(tokenData, &tok); err != nil {
		return nil, err
	}
	return cfg.TokenSource(context.Background(), &tok), nil
}

func fetchBills() string {
	ts, err := tokenSource()
	if err != nil {
		return fmt.Sprintf("(Google OAuth not configured: %v)", err)
	}
	ctx := context.Background()
	drv, err := drive.NewService(ctx, option.WithTokenSource(ts))
	if err != nil {
		return fmt.Sprintf("(Drive error: %v)", err)
	}
	q := fmt.Sprintf("name='%s' and mimeType='application/vnd.google-apps.spreadsheet'", config.BillsSheetName)
	res, err := drv.Files.List().Q(q).Fields("files(id)").PageSize(1).Do()
	if err != nil || len(res.Files) == 0 {
		return fmt.Sprintf("(No spreadsheet named '%s' found)", config.BillsSheetName)
	}
	svc, err := sheets.NewService(ctx, option.WithTokenSource(ts))
	if err != nil {
		return fmt.Sprintf("(Sheets error: %v)", err)
	}
	data, err := svc.Spreadsheets.Values.Get(res.Files[0].Id, "A1:Z500").Do()
	if err != nil || len(data.Values) == 0 {
		return fmt.Sprintf("(%s sheet empty or error: %v)", config.BillsSheetName, err)
	}
	var rows []string
	for _, row := range data.Values {
		cells := make([]string, len(row))
		for i, c := range row {
			cells[i] = fmt.Sprintf("%v", c)
		}
		rows = append(rows, strings.Join(cells, "\t"))
	}
	return config.BillsSheetName + " spreadsheet:\n" + strings.Join(rows, "\n")
}

func fetchGmail(query string) string {
	ts, err := tokenSource()
	if err != nil {
		return ""
	}
	ctx := context.Background()
	svc, err := gmail.NewService(ctx, option.WithTokenSource(ts))
	if err != nil {
		return fmt.Sprintf("(Gmail error: %v)", err)
	}
	if len(query) > 150 {
		query = query[:150]
	}
	res, err := svc.Users.Messages.List("me").Q(query).MaxResults(5).Do()
	if err != nil || len(res.Messages) == 0 {
		return "(No matching emails)"
	}
	var out []string
	for _, m := range res.Messages {
		d, err := svc.Users.Messages.Get("me", m.Id).
			Format("metadata").MetadataHeaders("Subject", "From", "Date").Do()
		if err != nil {
			continue
		}
		h := map[string]string{}
		for _, hdr := range d.Payload.Headers {
			h[hdr.Name] = hdr.Value
		}
		out = append(out, fmt.Sprintf("[%s] %s — from %s | %s", h["Date"], h["Subject"], h["From"], d.Snippet))
	}
	return "Recent emails:\n" + strings.Join(out, "\n")
}

// BuildContext fetches Google context (Bills sheet and/or Gmail) relevant to the prompt.
func BuildContext(prompt string) string {
	lower := strings.ToLower(prompt)
	var parts []string
	if containsAny(lower, config.FinanceKeywords) {
		if s := fetchBills(); s != "" {
			parts = append(parts, s)
		}
	}
	if containsAny(lower, config.EmailTerms) {
		if s := fetchGmail(prompt); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n\n")
}

func containsAny(s string, keywords []string) bool {
	for _, kw := range keywords {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func toContents(history []ollama.Msg) []*genai.Content {
	var contents []*genai.Content
	for _, m := range history {
		if m.Content == "" {
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: []genai.Part{genai.Text(m.Content)},
		})
	}
	return contents
}

func newModel(apiKey, system string) (*genai.GenerativeModel, *genai.Client, error) {
	client, err := genai.NewClient(context.Background(), option.WithAPIKey(apiKey))
	if err != nil {
		return nil, nil, err
	}
	m := client.GenerativeModel("gemini-2.0-flash")
	m.SystemInstruction = &genai.Content{Parts: []genai.Part{genai.Text(system)}}
	return m, client, nil
}

// Chat sends a single non-streaming request to Gemini.
func Chat(prompt, system string, history []ollama.Msg) (string, error) {
	apiKey := GetAPIKey()
	if apiKey == "" {
		return "", fmt.Errorf("no Gemini API key — run 'hermes auth gemini' or set GEMINI_API_KEY")
	}
	if ctx := BuildContext(prompt); ctx != "" {
		system = system + "\n\n" + ctx
	}
	model, client, err := newModel(apiKey, system)
	if err != nil {
		return "", err
	}
	defer client.Close()

	session := model.StartChat()
	session.History = toContents(history)

	resp, err := session.SendMessage(context.Background(), genai.Text(prompt))
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, c := range resp.Candidates {
		for _, p := range c.Content.Parts {
			if t, ok := p.(genai.Text); ok {
				sb.WriteString(string(t))
			}
		}
	}
	return sb.String(), nil
}

// Stream returns a channel of text chunks from Gemini.
func Stream(prompt, system string, history []ollama.Msg) (<-chan string, error) {
	apiKey := GetAPIKey()
	if apiKey == "" {
		return nil, fmt.Errorf("no Gemini API key — run 'hermes auth gemini' or set GEMINI_API_KEY")
	}
	if ctx := BuildContext(prompt); ctx != "" {
		system = system + "\n\n" + ctx
	}
	model, client, err := newModel(apiKey, system)
	if err != nil {
		return nil, err
	}
	session := model.StartChat()
	session.History = toContents(history)

	ch := make(chan string, 32)
	go func() {
		defer client.Close()
		defer close(ch)
		iter := session.SendMessageStream(context.Background(), genai.Text(prompt))
		for {
			resp, err := iter.Next()
			if err != nil {
				break
			}
			for _, c := range resp.Candidates {
				for _, p := range c.Content.Parts {
					if t, ok := p.(genai.Text); ok && t != "" {
						ch <- string(t)
					}
				}
			}
		}
	}()
	return ch, nil
}
