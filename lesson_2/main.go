package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
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
	"sync"
	"time"
)

type responsesRequest struct {
	Model           string     `json:"model"`
	Input           string     `json:"input"`
	Instructions    string     `json:"instructions,omitempty"`
	Store           bool       `json:"store,omitempty"`
	Temperature     *float64   `json:"temperature,omitempty"`
	TopP            *float64   `json:"top_p,omitempty"`
	MaxOutputTokens *int       `json:"max_output_tokens,omitempty"`
	Text            *textBlock `json:"text,omitempty"`
	Stream          bool       `json:"stream,omitempty"`
}

type textBlock struct {
	Format map[string]any `json:"format,omitempty"`
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

type requestControl struct {
	Gen          generationParams
	Instructions string
	TextFormat   map[string]any
	StopMarker   string // not sent to API; used for client-side early stop / truncation
}

func (c requestControl) Summary() string {
	var b strings.Builder
	b.WriteString(c.Gen.Summary())
	if strings.TrimSpace(c.Instructions) != "" {
		b.WriteString(" instructions=on")
	} else {
		b.WriteString(" instructions=off")
	}
	if strings.TrimSpace(c.StopMarker) != "" {
		b.WriteString(" stop=marker")
	} else {
		b.WriteString(" stop=off")
	}
	formatType := "text"
	if c.TextFormat != nil {
		if t, ok := c.TextFormat["type"].(string); ok && strings.TrimSpace(t) != "" {
			formatType = t
		}
	}
	b.WriteString(" format=")
	b.WriteString(formatType)
	return b.String()
}

func callChatGPT(ctx context.Context, client *http.Client, apiKey string, control requestControl, prompt string) (string, error) {
	if strings.TrimSpace(apiKey) == "" {
		return "", errors.New("missing OPENAI_API_KEY")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", errors.New("prompt is empty")
	}
	if strings.TrimSpace(control.Gen.Model) == "" {
		control.Gen.Model = "gpt-5.2"
	}
	if err := validateParams(control.Gen); err != nil {
		return "", err
	}

	temp := control.Gen.Temperature
	topP := control.Gen.TopP
	var maxOut *int
	if control.Gen.MaxTokens > 0 {
		v := control.Gen.MaxTokens
		maxOut = &v
	}

	reqBody, err := json.Marshal(responsesRequest{
		Model:           control.Gen.Model,
		Input:           prompt,
		Instructions:    strings.TrimSpace(control.Instructions),
		Store:           false,
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: maxOut,
		Text: func() *textBlock {
			if control.TextFormat == nil {
				return nil
			}
			return &textBlock{Format: control.TextFormat}
		}(),
	})
	if err != nil {
		return "", err
	}

	newRequest := func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")
		return req, nil
	}

	const maxRetries = 2
	for attempt := 0; ; attempt++ {
		req, err := newRequest()
		if err != nil {
			return "", err
		}
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
		if m := strings.TrimSpace(control.StopMarker); m != "" {
			if idx := strings.Index(out, m); idx >= 0 {
				out = strings.TrimSpace(out[:idx])
			}
		}
		return out, nil
	}
}

func callChatGPTStream(ctx context.Context, client *http.Client, apiKey string, control requestControl, prompt string, onDelta func(string)) error {
	if strings.TrimSpace(apiKey) == "" {
		return errors.New("missing OPENAI_API_KEY")
	}
	if strings.TrimSpace(prompt) == "" {
		return errors.New("prompt is empty")
	}
	if strings.TrimSpace(control.Gen.Model) == "" {
		control.Gen.Model = "gpt-5.2"
	}
	if err := validateParams(control.Gen); err != nil {
		return err
	}

	temp := control.Gen.Temperature
	topP := control.Gen.TopP
	var maxOut *int
	if control.Gen.MaxTokens > 0 {
		v := control.Gen.MaxTokens
		maxOut = &v
	}

	reqBody, err := json.Marshal(responsesRequest{
		Model:           control.Gen.Model,
		Input:           prompt,
		Instructions:    strings.TrimSpace(control.Instructions),
		Store:           false,
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: maxOut,
		Text: func() *textBlock {
			if control.TextFormat == nil {
				return nil
			}
			return &textBlock{Format: control.TextFormat}
		}(),
		Stream: true,
	})
	if err != nil {
		return err
	}

	newRequest := func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(reqBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		return req, nil
	}

	const maxRetries = 2
	for attempt := 0; ; attempt++ {
		req, err := newRequest()
		if err != nil {
			return err
		}

		resp, err := client.Do(req)
		if err != nil {
			return err
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			raw, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				return readErr
			}
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
					return ctx.Err()
				case <-t.C:
				}
				continue
			}
			return err
		}

		// SSE format may send "event:" lines + JSON in "data:" lines.
		// Some payloads don't include "type" field; then the event name is the type.
		reader := bufio.NewReader(resp.Body)
		var (
			eventName  string
			dataLines  []string
			seenOutput bool
			emitted    strings.Builder
		)

		stopMarker := strings.TrimSpace(control.StopMarker)
		emit := func(d string) {
			if d == "" {
				return
			}
			emitted.WriteString(d)
			onDelta(d)
		}

		errStop := errors.New("stop marker reached")

		processEvent := func(eventName string, data string) error {
			data = strings.TrimSpace(data)
			if data == "" {
				return nil
			}
			if data == "[DONE]" {
				return io.EOF
			}

			var m map[string]any
			if err := json.Unmarshal([]byte(data), &m); err != nil {
				return nil
			}
			typ := strings.TrimSpace(eventName)
			if t, ok := m["type"].(string); ok && strings.TrimSpace(t) != "" {
				typ = t
			}

			extractDelta := func(m map[string]any) string {
				if d, ok := m["delta"].(string); ok && d != "" {
					return d
				}
				if d, ok := m["text"].(string); ok && d != "" {
					return d
				}
				if c, ok := m["content"].([]any); ok {
					var b strings.Builder
					for _, item := range c {
						obj, _ := item.(map[string]any)
						if obj == nil {
							continue
						}
						if t, _ := obj["type"].(string); t != "" && t != "output_text" && t != "output_text_delta" {
							continue
						}
						if d, ok := obj["delta"].(string); ok && d != "" {
							b.WriteString(d)
						}
						if txt, ok := obj["text"].(string); ok && txt != "" {
							b.WriteString(txt)
						}
					}
					return b.String()
				}
				return ""
			}

			if strings.Contains(typ, "output_text.delta") {
				if d := extractDelta(m); d != "" {
					seenOutput = true
					if stopMarker == "" {
						emit(d)
						return nil
					}
					// client-side stop: emit until marker and then stop
					combined := emitted.String() + d
					if idx := strings.Index(combined, stopMarker); idx >= 0 {
						remaining := combined[len(emitted.String()):idx]
						emit(remaining)
						return errStop
					}
					emit(d)
				}
				return nil
			}

			// Some streams emit non-delta output_text events (e.g. "response.output_text" / "response.output_text.done").
			if strings.Contains(typ, "output_text") {
				if d := extractDelta(m); d != "" {
					seenOutput = true
					if stopMarker == "" {
						emit(d)
						return nil
					}
					combined := emitted.String() + d
					if idx := strings.Index(combined, stopMarker); idx >= 0 {
						remaining := combined[len(emitted.String()):idx]
						emit(remaining)
						return errStop
					}
					emit(d)
				}
				return nil
			}

			// Some streams include the final response object in a "completed" event.
			if strings.Contains(typ, "response.completed") || strings.Contains(typ, "completed") {
				if seenOutput {
					return nil
				}
				var rr responsesResponse
				if err := json.Unmarshal([]byte(data), &rr); err != nil {
					return nil
				}
				if out := strings.TrimSpace(extractOutputText(rr)); out != "" {
					seenOutput = true
					if stopMarker != "" {
						if idx := strings.Index(out, stopMarker); idx >= 0 {
							out = strings.TrimSpace(out[:idx])
						}
					}
					emit(out)
				}
				return nil
			}

			if strings.Contains(typ, "error") {
				if e, ok := m["error"].(map[string]any); ok {
					if msg, ok := e["message"].(string); ok && strings.TrimSpace(msg) != "" {
						return errors.New(msg)
					}
				}
			}
			return nil
		}

		for {
			line, err := reader.ReadString('\n')
			if err != nil && !errors.Is(err, io.EOF) {
				_ = resp.Body.Close()
				return err
			}
			line = strings.TrimRight(line, "\r\n")

			if line == "" {
				data := strings.Join(dataLines, "\n")
				dataLines = dataLines[:0]
				err2 := processEvent(eventName, data)
				eventName = ""
				if errors.Is(err2, errStop) {
					_ = resp.Body.Close()
					return nil
				}
				if errors.Is(err2, io.EOF) {
					_ = resp.Body.Close()
					return nil
				}
				if err2 != nil {
					_ = resp.Body.Close()
					return err2
				}
				if errors.Is(err, io.EOF) {
					_ = resp.Body.Close()
					return nil
				}
				continue
			}

			if strings.HasPrefix(line, "event:") {
				eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			}
			if strings.HasPrefix(line, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}

			if errors.Is(err, io.EOF) {
				// flush any buffered event at EOF
				data := strings.Join(dataLines, "\n")
				_ = resp.Body.Close()
				err2 := processEvent(eventName, data)
				if errors.Is(err2, errStop) {
					return nil
				}
				return err2
			}
		}
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
    <form method="POST">
      <div class="row">
        <label>Модель: <input type="text" name="model" value="{{.Model}}" /></label>
        <label>top_k: <input type="text" name="top_k" value="{{.TopK}}" /></label>
        <label>temperature: <input type="text" name="temperature" value="{{.Temperature}}" /></label>
        <label>top_p: <input type="text" name="top_p" value="{{.TopP}}" /></label>
        <label>max_tokens: <input type="text" name="max_tokens" value="{{.MaxTokens}}" /></label>
        <label>seed: <input type="text" name="seed" value="{{if ge .Seed 0}}{{.Seed}}{{end}}" /></label>
        <label>frequency_penalty: <input type="text" name="frequency_penalty" value="{{.FrequencyPenalty}}" /></label>
        <label>presence_penalty: <input type="text" name="presence_penalty" value="{{.PresencePenalty}}" /></label>
        <button type="submit" formaction="/ask">Отправить</button>
        <button type="submit" formaction="/compare">Сравнить</button>
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
    {{if .Compare}}
      <h2>Сравнение</h2>
      <h3>Без ограничений</h3>
      <pre>{{.Compare.UncontrolledParams}}</pre>
      <pre>{{.Compare.UncontrolledAnswer}}</pre>
      <h3>С ограничениями</h3>
      <pre>{{.Compare.ControlledParams}}</pre>
      <pre>{{.Compare.ControlledAnswer}}</pre>
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
	Compare          *compareData
}

type compareData struct {
	UncontrolledParams string
	UncontrolledAnswer string
	ControlledParams   string
	ControlledAnswer   string
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

type chatMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

type chatStore struct {
	mu       sync.Mutex
	sessions map[string][]chatMessage
	active   map[string]activeGen
}

type activeGen struct {
	id     string
	cancel context.CancelFunc
}

func (s *chatStore) get(sessionID string) []chatMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs := s.sessions[sessionID]
	out := make([]chatMessage, len(msgs))
	copy(out, msgs)
	return out
}

func (s *chatStore) append(sessionID string, msg chatMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[sessionID] = append(s.sessions[sessionID], msg)
}

func (s *chatStore) reset(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, sessionID)
	if g := s.active[sessionID]; g.cancel != nil {
		g.cancel()
	}
	delete(s.active, sessionID)
}

func (s *chatStore) setActive(sessionID string, cancel context.CancelFunc) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if prev := s.active[sessionID]; prev.cancel != nil {
		prev.cancel()
	}
	id, _ := newSessionID()
	s.active[sessionID] = activeGen{id: id, cancel: cancel}
	return id
}

func (s *chatStore) clearActive(sessionID string, id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if current, ok := s.active[sessionID]; ok && current.id == id {
		delete(s.active, sessionID)
	}
}

func (s *chatStore) stop(sessionID string) bool {
	s.mu.Lock()
	cancel := s.active[sessionID].cancel
	s.mu.Unlock()
	if cancel == nil {
		return false
	}
	cancel()
	return true
}

func newSessionID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b[:]), nil
}

func getSessionID(w http.ResponseWriter, r *http.Request) (string, error) {
	if c, err := r.Cookie("sid"); err == nil && strings.TrimSpace(c.Value) != "" {
		return c.Value, nil
	}
	sid, err := newSessionID()
	if err != nil {
		return "", err
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "sid",
		Value:    sid,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   60 * 60 * 24 * 30,
	})
	return sid, nil
}

func buildPrompt(history []chatMessage) string {
	const maxMsgs = 12
	if len(history) > maxMsgs {
		history = history[len(history)-maxMsgs:]
	}
	var b strings.Builder
	b.WriteString("You are a helpful assistant.\n\nConversation:\n")
	for _, m := range history {
		switch m.Role {
		case "user":
			b.WriteString("User: ")
		case "assistant":
			b.WriteString("Assistant: ")
		default:
			continue
		}
		b.WriteString(m.Text)
		b.WriteString("\n")
	}
	b.WriteString("Assistant:")
	return b.String()
}

type sendRequest struct {
	Message          string  `json:"message"`
	Model            string  `json:"model"`
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"top_p"`
	MaxTokens        int     `json:"max_tokens"`
	Compare          bool    `json:"compare"`
	Control          bool    `json:"control"`
	ControlMaxTokens int     `json:"control_max_tokens"`
}

type sseOut struct {
	Type    string `json:"type"`
	Variant string `json:"variant,omitempty"`
	Delta   string `json:"delta,omitempty"`
	Message string `json:"message,omitempty"`
}

func writeSSE(w http.ResponseWriter, f http.Flusher, v sseOut) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", b)
	if err != nil {
		return err
	}
	f.Flush()
	return nil
}

var chatPageTmpl = template.Must(template.New("chat").Parse(`<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Lesson 2 • Chat</title>
    <style>
      :root{
        --bg: #0b0f14;
        --panel: #0f1621;
        --panel2:#0c131d;
        --text: #e7eef9;
        --muted: #9fb0c7;
        --border: rgba(255,255,255,0.08);
        --user: #1f2a3a;
        --assistant: #111b29;
        --accent: #2f81f7;
      }
      html, body { height: 100%; margin: 0; }
      body { background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
      .app { height: 100%; display: flex; }
      .sidebar { width: 280px; background: var(--panel2); border-right: 1px solid var(--border); padding: 16px; box-sizing: border-box; display:flex; flex-direction:column; gap:12px;}
      .brand { font-weight: 700; letter-spacing: .2px; }
      .muted { color: var(--muted); font-size: 13px; line-height: 1.35; }
      .btn { background: rgba(255,255,255,0.06); color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; cursor: pointer; }
      .btn:hover { border-color: rgba(255,255,255,0.18); }
      .btn.primary { background: var(--accent); border-color: rgba(0,0,0,0); color: #fff; }
      .field { display:flex; flex-direction:column; gap:6px; }
      label { font-size: 12px; color: var(--muted); }
      input, textarea {
        background: rgba(255,255,255,0.04);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        outline: none;
      }
      input:focus, textarea:focus { border-color: rgba(47,129,247,0.6); }
      .row { display:flex; gap:10px; align-items:center; }
      .row > * { flex: 1 1 auto; }
      .main { flex: 1 1 auto; display:flex; flex-direction:column; }
      .topbar { padding: 14px 18px; border-bottom: 1px solid var(--border); background: var(--panel); display:flex; justify-content:space-between; align-items:center; }
      .topbar .title { font-weight: 600; }
      .chat { flex: 1 1 auto; overflow:auto; padding: 18px; box-sizing:border-box; }
      .msg { max-width: 820px; margin: 0 auto 14px auto; display:flex; gap: 10px; }
      .bubble { padding: 12px 14px; border-radius: 14px; border: 1px solid var(--border); white-space: pre-wrap; line-height: 1.45; }
      .msg.user { justify-content:flex-end; }
      .msg.user .bubble { background: var(--user); }
      .msg.assistant .bubble { background: var(--assistant); }
      .msg .meta { width: 110px; color: var(--muted); font-size: 12px; padding-top: 6px; }
      .msg.user .meta { text-align:right; }
      .composer { padding: 14px 18px; border-top: 1px solid var(--border); background: var(--panel); }
      .composer-inner { max-width: 860px; margin: 0 auto; display:flex; gap: 10px; align-items:flex-end; }
      .composer textarea { flex: 1 1 auto; min-height: 52px; max-height: 200px; resize: none; }
      .hint { max-width: 860px; margin: 10px auto 0 auto; color: var(--muted); font-size: 12px; }
      .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,0.03); color: var(--muted); }
      .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; border: 1px solid var(--border); border-bottom-width: 2px; border-radius: 8px; padding: 2px 6px; background: rgba(255,255,255,0.04);}
      details { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; background: rgba(255,255,255,0.03); }
      summary { cursor: pointer; color: var(--muted); }
      .small { font-size: 12px; color: var(--muted); }
      .error { color: #ffb4b4; }
      .variantbar { display:flex; align-items:center; gap:10px; padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid var(--border); }
      .variantbar .nav { width: 34px; height: 30px; border-radius: 10px; border: 1px solid var(--border); background: rgba(255,255,255,0.03); color: var(--text); cursor:pointer; }
      .variantbar .nav:hover { border-color: rgba(255,255,255,0.18); }
      .variantbar .count { color: var(--muted); font-size: 12px; min-width: 40px; text-align:center; }
      .variantbar .label { color: var(--muted); font-size: 12px; }
      .varianttext { white-space: pre-wrap; line-height: 1.45; }
    </style>
  </head>
  <body>
    <div class="app">
      <div class="sidebar">
        <div class="brand">Lesson 2</div>
        <div class="muted">Интерфейс в стиле ChatGPT + стриминг ответа. Ключ берётся из <code>OPENAI_API_KEY</code> (не уходит в браузер).</div>

        <div class="field">
          <label>Model</label>
          <input id="model" type="text" value="{{.DefaultModel}}" />
        </div>

        <div class="row">
          <div class="field">
            <label>temperature</label>
            <input id="temperature" type="text" value="{{.DefaultTemperature}}" />
          </div>
          <div class="field">
            <label>top_p</label>
            <input id="top_p" type="text" value="{{.DefaultTopP}}" />
          </div>
        </div>

        <details>
          <summary>Контроль ответа (урок)</summary>
          <div class="small" style="margin-top:10px;">Один и тот же запрос можно отправить <b>без ограничений</b> или <b>с ограничениями</b>: формат + длина + stop.</div>
          <div class="row" style="margin-top:10px;">
            <label style="flex: 0 0 auto;"><input id="compare" type="checkbox" /> compare (2 ответа)</label>
          </div>
          <div class="row" style="margin-top:10px;">
            <label style="flex: 0 0 auto;"><input id="control" type="checkbox" /> control (ограничения)</label>
          </div>
          <div class="row" style="margin-top:10px;">
            <div class="field">
              <label>max_tokens (controlled)</label>
              <input id="control_max_tokens" type="text" value="{{.DefaultControlMaxTokens}}" />
            </div>
          </div>
          <div class="small" style="margin-top:10px;">
            Если выбраны <b>compare</b> и <b>control</b>, ответ покажется как варианты одного сообщения (как в ChatGPT).
          </div>
        </details>

        <button id="reset" class="btn">Новый чат</button>
        <div id="status" class="small"></div>
      </div>

      <div class="main">
        <div class="topbar">
          <div class="title">Chat</div>
          <div class="pill"><span class="kbd">Enter</span> отправить • <span class="kbd">Shift</span>+<span class="kbd">Enter</span> новая строка</div>
        </div>
        <div id="chat" class="chat"></div>
        <div class="composer">
          <div class="composer-inner">
            <textarea id="input" placeholder="Напишите сообщение..."></textarea>
            <button id="send" class="btn primary">Send</button>
          </div>
          <div class="hint">Можно включить compare/control, чтобы увидеть один и тот же запрос с разным уровнем контроля ответа.</div>
        </div>
      </div>
    </div>

    <script>
      const chat = document.getElementById('chat');
      const input = document.getElementById('input');
      const sendBtn = document.getElementById('send');
      const resetBtn = document.getElementById('reset');
      const statusEl = document.getElementById('status');
      let currentES = null;
      let isGenerating = false;

      const el = (tag, cls, text) => {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text !== undefined) node.textContent = text;
        return node;
      };

      const scrollToBottom = () => { chat.scrollTop = chat.scrollHeight; };

      function addMessage(role, text, label) {
        const wrap = el('div', 'msg ' + role);
        const meta = el('div', 'meta', label || (role === 'user' ? 'You' : 'Assistant'));
        const bubble = el('div', 'bubble', text || '');
        if (role === 'user') {
          wrap.appendChild(bubble);
          wrap.appendChild(meta);
        } else {
          wrap.appendChild(meta);
          wrap.appendChild(bubble);
        }
        chat.appendChild(wrap);
        scrollToBottom();
        return bubble;
      }

      async function loadHistory() {
        const res = await fetch('/api/history');
        if (!res.ok) return;
        const data = await res.json();
        chat.innerHTML = '';
        for (const m of (data.messages || [])) {
          addMessage(m.role === 'user' ? 'user' : 'assistant', m.text, m.role === 'user' ? 'You' : 'Assistant');
        }
      }

      async function refreshHealth() {
        try {
          const res = await fetch('/api/health');
          if (!res.ok) return;
          const data = await res.json();
          if (!data.has_key) {
            setStatus('OPENAI_API_KEY не задан — ответы не придут', true);
          }
        } catch {}
      }

      function getSettings() {
        const model = document.getElementById('model').value.trim();
        const temperature = parseFloat(document.getElementById('temperature').value.trim() || '1');
        const top_p = parseFloat(document.getElementById('top_p').value.trim() || '1');
        const compare = document.getElementById('compare').checked;
        const control = document.getElementById('control').checked;
        const control_max_tokens = parseInt(document.getElementById('control_max_tokens').value.trim() || '120', 10);
        return { model, temperature, top_p, compare, control, control_max_tokens };
      }

      function setStatus(text, isError) {
        statusEl.textContent = text || '';
        statusEl.className = 'small' + (isError ? ' error' : '');
      }

      function addVariantMessage() {
        const wrap = el('div', 'msg assistant');
        const meta = el('div', 'meta', 'Assistant');
        const bubble = el('div', 'bubble');

        const bar = el('div', 'variantbar');
        const prev = el('button', 'nav', '◀');
        const next = el('button', 'nav', '▶');
        const count = el('div', 'count', '1/2');
        const label = el('div', 'label', 'Без ограничений');
        const text = el('div', 'varianttext', '...');
        bar.appendChild(prev);
        bar.appendChild(count);
        bar.appendChild(next);
        bar.appendChild(label);
        bubble.appendChild(bar);
        bubble.appendChild(text);

        wrap.appendChild(meta);
        wrap.appendChild(bubble);
        chat.appendChild(wrap);
        scrollToBottom();

        const state = {
          selected: 'uncontrolled',
          buffers: { uncontrolled: '', controlled: '' },
          label,
          count,
          text,
          setSelected: function(variant) {
            this.selected = variant;
            if (variant === 'controlled') {
              this.count.textContent = '2/2';
              this.label.textContent = 'С ограничениями';
              this.text.textContent = this.buffers.controlled || '...';
            } else {
              this.count.textContent = '1/2';
              this.label.textContent = 'Без ограничений';
              this.text.textContent = this.buffers.uncontrolled || '...';
            }
            scrollToBottom();
          },
          append: function(variant, delta) {
            this.buffers[variant] += delta;
            if (this.selected === variant) {
              this.text.textContent = this.buffers[variant];
              scrollToBottom();
            }
          }
        };

        prev.addEventListener('click', () => state.setSelected(state.selected === 'controlled' ? 'uncontrolled' : 'uncontrolled'));
        next.addEventListener('click', () => state.setSelected(state.selected === 'uncontrolled' ? 'controlled' : 'controlled'));

        return state;
      }

      async function sendMessage() {
        if (isGenerating) return;
        const msg = input.value.trim();
        if (!msg) return;
        input.value = '';

        addMessage('user', msg);
        setStatus('Generating...', false);
        isGenerating = true;
        sendBtn.textContent = 'Stop';
        sendBtn.classList.remove('primary');

        const settings = getSettings();
        const q = new URLSearchParams();
        q.set('message', msg);
        q.set('model', settings.model);
        q.set('temperature', String(settings.temperature));
        q.set('top_p', String(settings.top_p));
        q.set('compare', settings.compare ? '1' : '0');
        q.set('control', settings.control ? '1' : '0');
        q.set('control_max_tokens', String(settings.control_max_tokens));

        const es = new EventSource('/api/send_sse?' + q.toString());
        currentES = es;
        const expected = (settings.compare ? ['uncontrolled', 'controlled'] : (settings.control ? ['controlled'] : ['uncontrolled']));
        const doneSet = new Set();
        const variantsUI = (settings.compare && settings.control);
        let seenDoneAll = false;
        let receivedAnyDelta = false;
        let bubbles = { uncontrolled: null, controlled: null };
        let variantMsg = null;
        if (variantsUI) {
          variantMsg = addVariantMessage();
        } else {
          bubbles.uncontrolled = addMessage('assistant', '...', 'Assistant');
        }

        es.onmessage = (e) => {
          let ev = null;
          try { ev = JSON.parse(e.data); } catch { return; }
          if (ev.type === 'begin') {
            const variant = ev.variant || 'uncontrolled';
            if (!variantsUI) {
              const label = variant === 'controlled' ? 'Assistant (controlled)' : 'Assistant';
              if (!bubbles[variant]) {
                bubbles[variant] = addMessage('assistant', '', label);
              } else {
                bubbles[variant].textContent = '';
              }
            } else {
              // default selected is uncontrolled; no-op
            }
            return;
          }
          if (ev.type === 'delta') {
            const variant = ev.variant || 'uncontrolled';
            receivedAnyDelta = true;
            if (variantsUI) {
              variantMsg.append(variant, ev.delta || '');
              return;
            }
            if (!bubbles[variant]) {
              const label = variant === 'controlled' ? 'Assistant (controlled)' : 'Assistant';
              bubbles[variant] = addMessage('assistant', '', label);
            }
            bubbles[variant].textContent += ev.delta || '';
            scrollToBottom();
            return;
          }
          if (ev.type === 'done') {
            doneSet.add(ev.variant || 'uncontrolled');
            return;
          }
          if (ev.type === 'done_all') {
            seenDoneAll = true;
            es.close();
            currentES = null;
            setStatus('', false);
            isGenerating = false;
            sendBtn.textContent = 'Send';
            sendBtn.classList.add('primary');
            return;
          }
          if (ev.type === 'error') {
            es.close();
            currentES = null;
            addMessage('assistant', ev.message || 'Unknown error', 'Error');
            setStatus('Error', true);
            isGenerating = false;
            sendBtn.textContent = 'Send';
            sendBtn.classList.add('primary');
            return;
          }
        };
        es.onerror = async () => {
          // EventSource fires onerror on normal close too; don't show scary message if we already finished.
          if (!isGenerating || seenDoneAll || doneSet.size >= expected.length) {
            try { es.close(); } catch {}
            currentES = null;
            isGenerating = false;
            sendBtn.textContent = 'Send';
            sendBtn.classList.add('primary');
            setStatus('', false);
            return;
          }
          // If we received some output, treat disconnect as stop/finish.
          if (receivedAnyDelta) {
            try { es.close(); } catch {}
            currentES = null;
            isGenerating = false;
            sendBtn.textContent = 'Send';
            sendBtn.classList.add('primary');
            setStatus('Disconnected', true);
            return;
          }
          try { es.close(); } catch {}
          currentES = null;
          setStatus('Error', true);
          addMessage('assistant', 'Streaming connection failed. Проверьте, что сервер запущен и OPENAI_API_KEY задан.', 'Error');
          isGenerating = false;
          sendBtn.textContent = 'Send';
          sendBtn.classList.add('primary');
        };
      }

      async function stopGeneration() {
        try { await fetch('/api/stop', { method: 'POST' }); } catch {}
        if (currentES) {
          currentES.close();
          currentES = null;
        }
        setStatus('Stopped', false);
        isGenerating = false;
        sendBtn.textContent = 'Send';
        sendBtn.classList.add('primary');
      }

      sendBtn.addEventListener('click', () => {
        if (isGenerating) return stopGeneration();
        return sendMessage();
      });
      resetBtn.addEventListener('click', async () => {
        await fetch('/api/reset', { method: 'POST' });
        await loadHistory();
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });

      loadHistory();
      refreshHealth();
    </script>
  </body>
</html>`))

type chatPageData struct {
	DefaultModel            string
	DefaultTemperature      string
	DefaultTopP             string
	DefaultControlMaxTokens string
	DefaultControlStop      string
}

func runServer(addr, defaultModel string) error {
	apiKey := os.Getenv("OPENAI_API_KEY")
	client := &http.Client{Timeout: 90 * time.Second}

	store := &chatStore{sessions: map[string][]chatMessage{}, active: map[string]activeGen{}}

	const (
		defaultControlMaxTokens = 120
		defaultControlStop      = "###END###"
	)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		_, _ = getSessionID(w, r)
		_ = chatPageTmpl.Execute(w, chatPageData{
			DefaultModel:            defaultModel,
			DefaultTemperature:      "1",
			DefaultTopP:             "1",
			DefaultControlMaxTokens: strconv.Itoa(defaultControlMaxTokens),
			DefaultControlStop:      defaultControlStop,
		})
	})

	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		_, _ = getSessionID(w, r)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"has_key": strings.TrimSpace(apiKey) != "",
		})
	})

	mux.HandleFunc("/api/history", func(w http.ResponseWriter, r *http.Request) {
		sid, err := getSessionID(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		msgs := store.get(sid)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"messages": msgs})
	})

	mux.HandleFunc("/api/reset", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		sid, err := getSessionID(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		store.reset(sid)
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/api/stop", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		sid, err := getSessionID(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = store.stop(sid)
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/api/send", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if strings.TrimSpace(apiKey) == "" {
			http.Error(w, "missing OPENAI_API_KEY", http.StatusBadRequest)
			return
		}
		sid, err := getSessionID(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var req sendRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		msg := strings.TrimSpace(req.Message)
		if msg == "" {
			http.Error(w, "message is empty", http.StatusBadRequest)
			return
		}

		model := strings.TrimSpace(req.Model)
		if model == "" {
			model = defaultModel
		}
		temperature := req.Temperature
		if temperature == 0 {
			temperature = 1
		}
		topP := req.TopP
		if topP == 0 {
			topP = 1
		}

		store.append(sid, chatMessage{Role: "user", Text: msg})
		history := store.get(sid)
		prompt := buildPrompt(history)

		log.Printf("/api/send sid=%s model=%s compare=%v control=%v", sid, model, req.Compare, req.Control)

		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		f, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		activeID := store.setActive(sid, cancel)
		defer func() {
			store.clearActive(sid, activeID)
			cancel()
		}()

		baseGen := generationParams{
			Model:       model,
			Temperature: temperature,
			TopP:        topP,
			TopK:        0,
			MaxTokens:   req.MaxTokens,
		}

		makeControlled := func() requestControl {
			maxT := req.ControlMaxTokens
			if maxT <= 0 {
				maxT = defaultControlMaxTokens
			}
			gen := baseGen
			gen.MaxTokens = maxT
			instructions := strings.Join([]string{
				"Ответь в формате списка из 5 пунктов.",
				"Каждый пункт — не более 12 слов.",
				"Не добавляй вступление и заключение.",
				fmt.Sprintf("В конце напиши маркер %s и остановись.", defaultControlStop),
			}, " ")
			return requestControl{Gen: gen, Instructions: instructions, StopMarker: defaultControlStop}
		}

		variantOrder := []string{}
		if req.Compare {
			variantOrder = []string{"uncontrolled", "controlled"}
		} else if req.Control {
			variantOrder = []string{"controlled"}
		} else {
			variantOrder = []string{"uncontrolled"}
		}

		var uncontrolledAnswer strings.Builder
		var singleAnswer strings.Builder

		runVariant := func(variant string) error {
			if err := writeSSE(w, f, sseOut{Type: "begin", Variant: variant}); err != nil {
				return err
			}
			var control requestControl
			switch variant {
			case "controlled":
				control = makeControlled()
			default:
				control = requestControl{Gen: baseGen}
			}
			var b strings.Builder
			err := callChatGPTStream(ctx, client, apiKey, control, prompt, func(d string) {
				b.WriteString(d)
				_ = writeSSE(w, f, sseOut{Type: "delta", Variant: variant, Delta: d})
			})
			if err != nil {
				return err
			}
			if err := writeSSE(w, f, sseOut{Type: "done", Variant: variant}); err != nil {
				return err
			}
			if variant == "uncontrolled" {
				uncontrolledAnswer.WriteString(b.String())
			}
			if len(variantOrder) == 1 {
				singleAnswer.WriteString(b.String())
			}
			return nil
		}

		for _, variant := range variantOrder {
			if err := runVariant(variant); err != nil {
				log.Printf("/api/send error sid=%s: %v", sid, err)
				_ = writeSSE(w, f, sseOut{Type: "error", Message: err.Error()})
				return
			}
		}

		// persist assistant message (keep history simple; in compare-mode store only the uncontrolled answer)
		if req.Compare {
			store.append(sid, chatMessage{Role: "assistant", Text: strings.TrimSpace(uncontrolledAnswer.String())})
		} else {
			store.append(sid, chatMessage{Role: "assistant", Text: strings.TrimSpace(singleAnswer.String())})
		}
	})

	mux.HandleFunc("/api/send_sse", func(w http.ResponseWriter, r *http.Request) {
		// EventSource requires GET.
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if strings.TrimSpace(apiKey) == "" {
			http.Error(w, "missing OPENAI_API_KEY", http.StatusBadRequest)
			return
		}
		sid, err := getSessionID(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		msg := strings.TrimSpace(r.URL.Query().Get("message"))
		if msg == "" {
			http.Error(w, "message is empty", http.StatusBadRequest)
			return
		}
		model := strings.TrimSpace(r.URL.Query().Get("model"))
		if model == "" {
			model = defaultModel
		}
		temperature := parseFloatDefault(r.URL.Query().Get("temperature"), 1)
		topP := parseFloatDefault(r.URL.Query().Get("top_p"), 1)
		compare := r.URL.Query().Get("compare") == "1"
		controlOnly := r.URL.Query().Get("control") == "1"
		controlMaxTokens := parseIntDefault(r.URL.Query().Get("control_max_tokens"), defaultControlMaxTokens)
		controlStop := defaultControlStop

		store.append(sid, chatMessage{Role: "user", Text: msg})
		history := store.get(sid)
		prompt := buildPrompt(history)

		log.Printf("/api/send_sse sid=%s model=%s compare=%v control=%v", sid, model, compare, controlOnly)

		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		f, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
		activeID := store.setActive(sid, cancel)
		defer func() {
			store.clearActive(sid, activeID)
			cancel()
		}()

		baseGen := generationParams{
			Model:       model,
			Temperature: temperature,
			TopP:        topP,
			TopK:        0,
			MaxTokens:   0,
		}

		makeControlled := func() requestControl {
			gen := baseGen
			gen.MaxTokens = controlMaxTokens
			instructions := strings.Join([]string{
				"Ответь в формате списка из 5 пунктов.",
				"Каждый пункт — не более 12 слов.",
				"Не добавляй вступление и заключение.",
				fmt.Sprintf("В конце напиши маркер %s и остановись.", controlStop),
			}, " ")
			return requestControl{Gen: gen, Instructions: instructions, StopMarker: controlStop}
		}

		variantOrder := []string{}
		if compare {
			variantOrder = []string{"uncontrolled", "controlled"}
		} else if controlOnly {
			variantOrder = []string{"controlled"}
		} else {
			variantOrder = []string{"uncontrolled"}
		}

		var uncontrolledAnswer strings.Builder
		var singleAnswer strings.Builder

		runVariant := func(variant string) error {
			if err := writeSSE(w, f, sseOut{Type: "begin", Variant: variant}); err != nil {
				return err
			}
			var control requestControl
			switch variant {
			case "controlled":
				control = makeControlled()
			default:
				control = requestControl{Gen: baseGen}
			}
			var b strings.Builder
			err := callChatGPTStream(ctx, client, apiKey, control, prompt, func(d string) {
				b.WriteString(d)
				_ = writeSSE(w, f, sseOut{Type: "delta", Variant: variant, Delta: d})
			})
			if err != nil {
				return err
			}
			if err := writeSSE(w, f, sseOut{Type: "done", Variant: variant}); err != nil {
				return err
			}
			if variant == "uncontrolled" {
				uncontrolledAnswer.WriteString(b.String())
			}
			if len(variantOrder) == 1 {
				singleAnswer.WriteString(b.String())
			}
			return nil
		}

		for _, variant := range variantOrder {
			if err := runVariant(variant); err != nil {
				log.Printf("/api/send_sse error sid=%s: %v", sid, err)
				_ = writeSSE(w, f, sseOut{Type: "error", Message: err.Error()})
				return
			}
		}

		_ = writeSSE(w, f, sseOut{Type: "done_all"})

		if compare {
			store.append(sid, chatMessage{Role: "assistant", Text: strings.TrimSpace(uncontrolledAnswer.String())})
		} else {
			store.append(sid, chatMessage{Role: "assistant", Text: strings.TrimSpace(singleAnswer.String())})
		}
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
		mode             = flag.String("mode", "cli", "cli | compare | serve")
		model            = flag.String("model", "gpt-5.2", "OpenAI model")
		topK             = flag.Int("top_k", 0, "sampling top_k (compat only; ignored by official OpenAI API)")
		temperature      = flag.Float64("temperature", 1, "sampling temperature (0..2)")
		topP             = flag.Float64("top_p", 1, "nucleus sampling top_p (0..1)")
		maxTokens        = flag.Int("max_tokens", 0, "max output tokens (sent as max_output_tokens for Responses API)")
		seed             = flag.Int64("seed", -1, "seed (not supported by Responses API; printed for tracking only)")
		freqPenalty      = flag.Float64("frequency_penalty", 0, "frequency penalty (-2..2) (not supported by Responses API; printed for tracking only)")
		presPenalty      = flag.Float64("presence_penalty", 0, "presence penalty (-2..2) (not supported by Responses API; printed for tracking only)")
		prompt           = flag.String("prompt", "", "prompt (CLI mode)")
		addr             = flag.String("addr", "127.0.0.1:8080", "listen address (serve mode)")
		controlMaxTokens = flag.Int("control_max_tokens", 120, "compare-mode: max output tokens for the controlled request")
		controlStop      = flag.String("control_stop", "###END###", "compare-mode: stop sequence for the controlled request")
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
		gen := generationParams{
			Model:            strings.TrimSpace(*model),
			Temperature:      *temperature,
			TopP:             *topP,
			TopK:             *topK,
			MaxTokens:        *maxTokens,
			Seed:             *seed,
			FrequencyPenalty: *freqPenalty,
			PresencePenalty:  *presPenalty,
		}

		control := requestControl{Gen: gen}
		fmt.Fprintln(os.Stderr, control.Summary())
		if *topK != 0 {
			fmt.Fprintln(os.Stderr, "note: top_k is ignored by official OpenAI API (use top_p/temperature instead).")
		}
		if *seed >= 0 || *freqPenalty != 0 || *presPenalty != 0 {
			fmt.Fprintln(os.Stderr, "note: seed/frequency_penalty/presence_penalty are not supported by the Responses API and are printed for tracking only.")
		}

		answer, err := callChatGPT(ctx, client, apiKey, control, p)
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
	case "compare":
		apiKey := os.Getenv("OPENAI_API_KEY")
		client := &http.Client{Timeout: 90 * time.Second}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		p := strings.TrimSpace(*prompt)
		if p == "" {
			raw, _ := io.ReadAll(os.Stdin)
			p = strings.TrimSpace(string(raw))
		}

		baseGen := generationParams{
			Model:       strings.TrimSpace(*model),
			Temperature: *temperature,
			TopP:        *topP,
			TopK:        *topK,
			MaxTokens:   0,
		}

		fmt.Fprintln(os.Stderr, "=== Without restrictions ===")
		uncontrolled := requestControl{Gen: baseGen}
		fmt.Fprintln(os.Stderr, uncontrolled.Summary())
		uncontrolledAnswer, err := callChatGPT(ctx, client, apiKey, uncontrolled, p)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Println("### Without restrictions")
		fmt.Println(uncontrolledAnswer)

		fmt.Fprintln(os.Stderr, "\n=== With restrictions ===")
		controlledGen := baseGen
		controlledGen.MaxTokens = *controlMaxTokens
		stop := strings.TrimSpace(*controlStop)
		if stop == "" {
			stop = "###END###"
		}
		controlledInstructions := strings.Join([]string{
			"Ответь строго в формате JSON (один объект).",
			"JSON должен содержать ключи: answer (string) и note (string).",
			"Не добавляй никакого текста вне JSON.",
			fmt.Sprintf("В конце напиши маркер %s и остановись.", stop),
		}, " ")
		controlled := requestControl{
			Gen:          controlledGen,
			Instructions: controlledInstructions,
			StopMarker:   stop,
			TextFormat:   map[string]any{"type": "json_object"},
		}
		fmt.Fprintln(os.Stderr, controlled.Summary())
		controlledAnswer, err := callChatGPT(ctx, client, apiKey, controlled, p)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
			os.Exit(1)
		}
		fmt.Println("\n### With restrictions")
		fmt.Println(controlledAnswer)
	default:
		log.Fatalf("unknown -mode=%q (use cli or serve)", *mode)
	}
}
