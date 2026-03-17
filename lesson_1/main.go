package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type responsesRequest struct {
	Model           string   `json:"model"`
	Input           string   `json:"input"`
	Store           bool     `json:"store,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
}

type responsesResponse struct {
	Output []struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

type openAIError struct {
	StatusCode int
	Status     string
	Message    string
	Type       string
	Code       string
}

func (e *openAIError) Error() string {
	if e == nil {
		return "openai error"
	}
	if e.StatusCode != 0 && e.Status != "" {
		return fmt.Sprintf("openai error (%s): %s", e.Status, e.Message)
	}
	if e.Message != "" {
		return "openai error: " + e.Message
	}
	return "openai error"
}

func isQuotaOrBilling429(err error) bool {
	var oe *openAIError
	if !errors.As(err, &oe) || oe == nil {
		return false
	}
	if oe.StatusCode != http.StatusTooManyRequests {
		return false
	}

	code := strings.ToLower(strings.TrimSpace(oe.Code))
	if code == "insufficient_quota" {
		return true
	}
	msg := strings.ToLower(oe.Message)
	return strings.Contains(msg, "exceeded your current quota") ||
		strings.Contains(msg, "check your plan") ||
		strings.Contains(msg, "billing")
}

func extractOutputText(r responsesResponse) string {
	var b strings.Builder
	for _, out := range r.Output {
		if out.Type != "message" || out.Role != "assistant" {
			continue
		}
		for _, c := range out.Content {
			if c.Type != "output_text" {
				continue
			}
			b.WriteString(c.Text)
		}
	}
	return b.String()
}

type generationParams struct {
	Model            string
	Temperature      float64
	TopP             float64
	TopK             int
	MaxTokens        int
	Seed             int64
	FrequencyPenalty float64
	PresencePenalty  float64
}

func (p generationParams) Summary() string {
	maxTokens := p.MaxTokens
	if maxTokens < 0 {
		maxTokens = 0
	}
	seedStr := "unset"
	if p.Seed >= 0 {
		seedStr = strconv.FormatInt(p.Seed, 10)
	}
	return fmt.Sprintf(
		"model=%s temperature=%g top_p=%g top_k=%d max_tokens=%d seed=%s frequency_penalty=%g presence_penalty=%g",
		strings.TrimSpace(p.Model),
		p.Temperature,
		p.TopP,
		p.TopK,
		maxTokens,
		seedStr,
		p.FrequencyPenalty,
		p.PresencePenalty,
	)
}

func validateParams(p generationParams) error {
	if strings.TrimSpace(p.Model) == "" {
		return errors.New("model is empty")
	}
	if p.Temperature < 0 || p.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2 (got %g)", p.Temperature)
	}
	if p.TopP < 0 || p.TopP > 1 {
		return fmt.Errorf("top_p must be between 0 and 1 (got %g)", p.TopP)
	}
	if p.MaxTokens < 0 {
		return fmt.Errorf("max_tokens must be >= 0 (got %d)", p.MaxTokens)
	}
	if p.TopK < 0 {
		return fmt.Errorf("top_k must be >= 0 (got %d)", p.TopK)
	}
	if p.FrequencyPenalty < -2 || p.FrequencyPenalty > 2 {
		return fmt.Errorf("frequency_penalty must be between -2 and 2 (got %g)", p.FrequencyPenalty)
	}
	if p.PresencePenalty < -2 || p.PresencePenalty > 2 {
		return fmt.Errorf("presence_penalty must be between -2 and 2 (got %g)", p.PresencePenalty)
	}
	return nil
}

func callChatGPT(ctx context.Context, client *http.Client, apiKey string, params generationParams, prompt string) (string, error) {
	if strings.TrimSpace(apiKey) == "" {
		return "", errors.New("missing OPENAI_API_KEY")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", errors.New("prompt is empty")
	}
	if strings.TrimSpace(params.Model) == "" {
		params.Model = "gpt-5.2"
	}
	if err := validateParams(params); err != nil {
		return "", err
	}

	temp := params.Temperature
	topP := params.TopP
	var maxOut *int
	if params.MaxTokens > 0 {
		v := params.MaxTokens
		maxOut = &v
	}

	reqBody, err := json.Marshal(responsesRequest{
		Model:           params.Model,
		Input:           prompt,
		Store:           false,
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: maxOut,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	const maxRetries = 2
	for attempt := 0; ; attempt++ {
		resp, err := client.Do(req)
		if err != nil {
			return "", err
		}

		raw, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return "", readErr
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			var apiErr responsesResponse
			_ = json.Unmarshal(raw, &apiErr)

			msg := strings.TrimSpace(string(raw))
			typ := ""
			code := ""
			if apiErr.Error != nil {
				if apiErr.Error.Message != "" {
					msg = apiErr.Error.Message
				}
				typ = apiErr.Error.Type
				code = apiErr.Error.Code
			}
			err := &openAIError{
				StatusCode: resp.StatusCode,
				Status:     resp.Status,
				Message:    msg,
				Type:       typ,
				Code:       code,
			}

			if resp.StatusCode == http.StatusTooManyRequests && !isQuotaOrBilling429(err) && attempt < maxRetries {
				backoff := time.Duration(500*(1<<attempt)) * time.Millisecond
				t := time.NewTimer(backoff)
				select {
				case <-ctx.Done():
					t.Stop()
					return "", ctx.Err()
				case <-t.C:
				}
				continue
			}
			return "", err
		}

		var rr responsesResponse
		if err := json.Unmarshal(raw, &rr); err != nil {
			return "", err
		}
		if rr.Error != nil && rr.Error.Message != "" {
			return "", &openAIError{
				StatusCode: resp.StatusCode,
				Status:     resp.Status,
				Message:    rr.Error.Message,
				Type:       rr.Error.Type,
				Code:       rr.Error.Code,
			}
		}

		out := strings.TrimSpace(extractOutputText(rr))
		if out == "" {
			out = strings.TrimSpace(string(raw))
		}
		return out, nil
	}
}

var pageTmpl = template.Must(template.New("page").Parse(`<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Local ChatGPT (Go)</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }
      textarea { width: 100%; min-height: 140px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      pre { white-space: pre-wrap; padding: 12px; background: #f6f8fa; border-radius: 8px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      .row > * { flex: 0 0 auto; }
      input[type="text"]{ width: 280px; }
      button { padding: 8px 14px; }
      .muted { color: #666; font-size: 14px; }
    </style>
  </head>
  <body>
    <h1>Local ChatGPT (Go)</h1>
    <form method="POST" action="/ask">
      <div class="row">
        <label>Модель: <input type="text" name="model" value="{{.Model}}" /></label>
        <label>top_k: <input type="text" name="top_k" value="{{.TopK}}" /></label>
        <label>temperature: <input type="text" name="temperature" value="{{.Temperature}}" /></label>
        <label>top_p: <input type="text" name="top_p" value="{{.TopP}}" /></label>
        <label>max_tokens: <input type="text" name="max_tokens" value="{{.MaxTokens}}" /></label>
        <label>seed: <input type="text" name="seed" value="{{if ge .Seed 0}}{{.Seed}}{{end}}" /></label>
        <label>frequency_penalty: <input type="text" name="frequency_penalty" value="{{.FrequencyPenalty}}" /></label>
        <label>presence_penalty: <input type="text" name="presence_penalty" value="{{.PresencePenalty}}" /></label>
        <button type="submit">Отправить</button>
      </div>
      <p class="muted">Ключ берётся из переменной окружения <code>OPENAI_API_KEY</code> (на клиент не отдаётся).</p>
      <p class="muted"><code>top_k</code> в официальном OpenAI API не поддерживается и тут показывается только как параметр совместимости (на запрос не влияет).</p>
      <p class="muted"><code>seed</code>, <code>frequency_penalty</code>, <code>presence_penalty</code> в Responses API не поддерживаются и тут выводятся только для трекинга.</p>
      <p><textarea name="prompt" placeholder="Введите вопрос...">{{.Prompt}}</textarea></p>
    </form>
    {{if .ParamsSummary}}
      <h2>Параметры</h2>
      <pre>{{.ParamsSummary}}</pre>
    {{end}}
    {{if .Answer}}
      <h2>Ответ</h2>
      <pre>{{.Answer}}</pre>
    {{end}}
    {{if .Error}}
      <h2>Ошибка</h2>
      <pre>{{.Error}}</pre>
    {{end}}
  </body>
</html>`))

type pageData struct {
	Model            string
	TopK             int
	Temperature      float64
	TopP             float64
	MaxTokens        int
	Seed             int64
	FrequencyPenalty float64
	PresencePenalty  float64
	ParamsSummary    string
	Prompt           string
	Answer           string
	Error            string
}

func parseFloatDefault(s string, def float64) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return def
	}
	return v
}

func parseIntDefault(s string, def int) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return v
}

func parseInt64Default(s string, def int64) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return def
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return def
	}
	return v
}

func runServer(addr, defaultModel string) error {
	apiKey := os.Getenv("OPENAI_API_KEY")
	client := &http.Client{Timeout: 90 * time.Second}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		_ = pageTmpl.Execute(w, pageData{
			Model:            defaultModel,
			TopK:             0,
			Temperature:      1,
			TopP:             1,
			MaxTokens:        0,
			Seed:             -1,
			FrequencyPenalty: 0,
			PresencePenalty:  0,
		})
	})
	mux.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Redirect(w, r, "/", http.StatusSeeOther)
			return
		}
		if err := r.ParseForm(); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = pageTmpl.Execute(w, pageData{Model: defaultModel, Error: err.Error()})
			return
		}
		model := strings.TrimSpace(r.FormValue("model"))
		if model == "" {
			model = defaultModel
		}
		topK := parseIntDefault(r.FormValue("top_k"), 0)
		temperature := parseFloatDefault(r.FormValue("temperature"), 1)
		topP := parseFloatDefault(r.FormValue("top_p"), 1)
		maxTokens := parseIntDefault(r.FormValue("max_tokens"), 0)
		seed := parseInt64Default(r.FormValue("seed"), -1)
		freqPenalty := parseFloatDefault(r.FormValue("frequency_penalty"), 0)
		presPenalty := parseFloatDefault(r.FormValue("presence_penalty"), 0)
		prompt := strings.TrimSpace(r.FormValue("prompt"))

		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()

		params := generationParams{
			Model:            model,
			Temperature:      temperature,
			TopP:             topP,
			TopK:             topK,
			MaxTokens:        maxTokens,
			Seed:             seed,
			FrequencyPenalty: freqPenalty,
			PresencePenalty:  presPenalty,
		}
		answer, err := callChatGPT(ctx, client, apiKey, params, prompt)
		data := pageData{
			Model:            model,
			TopK:             topK,
			Temperature:      temperature,
			TopP:             topP,
			MaxTokens:        maxTokens,
			Seed:             seed,
			FrequencyPenalty: freqPenalty,
			PresencePenalty:  presPenalty,
			ParamsSummary:    params.Summary(),
			Prompt:           prompt,
		}
		if err != nil {
			if isQuotaOrBilling429(err) {
				data.Error = err.Error() + "\n\nПохоже, у аккаунта/проекта сейчас нет доступной квоты. Проверьте Billing/Plan и Usage в OpenAI Dashboard."
			} else {
				data.Error = err.Error()
			}
		} else {
			data.Answer = answer
		}
		_ = pageTmpl.Execute(w, data)
	})

	s := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("Open http://%s\n", addr)
	return s.ListenAndServe()
}

func main() {
	var (
		mode        = flag.String("mode", "cli", "cli | serve")
		model       = flag.String("model", "gpt-5.2", "OpenAI model")
		topK        = flag.Int("top_k", 0, "sampling top_k (compat only; ignored by official OpenAI API)")
		temperature = flag.Float64("temperature", 1, "sampling temperature (0..2)")
		topP        = flag.Float64("top_p", 1, "nucleus sampling top_p (0..1)")
		maxTokens   = flag.Int("max_tokens", 0, "max output tokens (sent as max_output_tokens for Responses API)")
		seed        = flag.Int64("seed", -1, "seed (not supported by Responses API; printed for tracking only)")
		freqPenalty = flag.Float64("frequency_penalty", 0, "frequency penalty (-2..2) (not supported by Responses API; printed for tracking only)")
		presPenalty = flag.Float64("presence_penalty", 0, "presence penalty (-2..2) (not supported by Responses API; printed for tracking only)")
		prompt      = flag.String("prompt", "", "prompt (CLI mode)")
		addr        = flag.String("addr", "127.0.0.1:8080", "listen address (serve mode)")
	)
	flag.Parse()

	switch strings.ToLower(strings.TrimSpace(*mode)) {
	case "serve", "server", "web":
		if err := runServer(*addr, *model); err != nil {
			log.Fatal(err)
		}
	case "cli":
		apiKey := os.Getenv("OPENAI_API_KEY")
		client := &http.Client{Timeout: 90 * time.Second}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		p := strings.TrimSpace(*prompt)
		if p == "" {
			raw, _ := io.ReadAll(os.Stdin)
			p = strings.TrimSpace(string(raw))
		}
		params := generationParams{
			Model:            strings.TrimSpace(*model),
			Temperature:      *temperature,
			TopP:             *topP,
			TopK:             *topK,
			MaxTokens:        *maxTokens,
			Seed:             *seed,
			FrequencyPenalty: *freqPenalty,
			PresencePenalty:  *presPenalty,
		}

		fmt.Fprintln(os.Stderr, params.Summary())
		if *topK != 0 {
			fmt.Fprintln(os.Stderr, "note: top_k is ignored by official OpenAI API (use top_p/temperature instead).")
		}
		if *seed >= 0 || *freqPenalty != 0 || *presPenalty != 0 {
			fmt.Fprintln(os.Stderr, "note: seed/frequency_penalty/presence_penalty are not supported by the Responses API and are printed for tracking only.")
		}

		answer, err := callChatGPT(ctx, client, apiKey, params, p)
		if err != nil {
			if isQuotaOrBilling429(err) {
				fmt.Fprintln(os.Stderr, err.Error())
				fmt.Fprintln(os.Stderr, "Похоже, у аккаунта/проекта сейчас нет доступной квоты. Проверьте Billing/Plan и Usage в OpenAI Dashboard.")
				os.Exit(0)
			}
			var oe *openAIError
			if errors.As(err, &oe) && oe != nil && oe.StatusCode == http.StatusTooManyRequests {
				fmt.Fprintln(os.Stderr, err.Error())
				fmt.Fprintln(os.Stderr, "Похоже на rate limit (429). Попробуйте повторить позже или уменьшить частоту запросов.")
				os.Exit(0)
			}
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Println(answer)
	default:
		log.Fatalf("unknown -mode=%q (use cli or serve)", *mode)
	}
}
