package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

type Feature struct {
	Vocabulary map[string]int `json:"vocabulary"`
	IDF        []float64      `json:"idf"`
	Ngram      []int          `json:"ngram_range"`
	Lowercase  bool           `json:"lowercase"`
	Sublinear  bool           `json:"sublinear_tf"`
	Norm       string         `json:"norm"`
}
type Calibration struct {
	Method      string    `json:"method"`
	Coefficient float64   `json:"coefficient"`
	Intercept   float64   `json:"intercept"`
	X           []float64 `json:"x_thresholds"`
	Y           []float64 `json:"y_thresholds"`
}
type Policy struct {
	Threshold float64   `json:"positive_probability_threshold"`
	Review    []float64 `json:"review_probability_range"`
}
type Input struct {
	Fields []string `json:"fields"`
}
type Bundle struct {
	Format      int               `json:"format_version"`
	Version     string            `json:"model_version"`
	Task        map[string]string `json:"task"`
	Labels      []string          `json:"labels"`
	Positive    string            `json:"positive_class"`
	Probability string            `json:"positive_probability_field"`
	Input       Input             `json:"input"`
	Word        Feature           `json:"word"`
	Char        Feature           `json:"char"`
	Coef        [][]float64       `json:"coefficients"`
	Intercept   []float64         `json:"intercept"`
	Calibration Calibration       `json:"calibration"`
	Policy      Policy            `json:"decision_policy"`
}

func verify(path string) ([]byte, error) {
	raw, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	side, e := os.ReadFile(path + ".sha256")
	if e != nil {
		return nil, e
	}
	fields := strings.Fields(string(side))
	if len(fields) == 0 {
		return nil, errors.New("invalid bundle hash")
	}
	sum := sha256.Sum256(raw)
	if fields[0] != hex.EncodeToString(sum[:]) {
		return nil, errors.New("bundle hash mismatch")
	}
	return raw, nil
}
func load(path string) (Bundle, error) {
	raw, e := verify(path)
	if e != nil {
		return Bundle{}, e
	}
	var b Bundle
	e = json.Unmarshal(raw, &b)
	if e != nil {
		return b, e
	}
	if b.Format != 3 || len(b.Labels) != 2 || b.Positive == "" || b.Probability == "" || len(b.Input.Fields) == 0 || len(b.Coef) != 1 || len(b.Intercept) != 1 || len(b.Policy.Review) != 2 {
		return b, errors.New("unsupported artifact contract")
	}
	if b.Positive != b.Labels[0] && b.Positive != b.Labels[1] {
		return b, errors.New("unsupported artifact contract")
	}
	return b, nil
}
func words(s string) []string {
	out := []string{}
	r := []rune(strings.ToLower(s))
	start := -1
	for i, c := range r {
		if unicode.IsLetter(c) || unicode.IsDigit(c) || c == '_' {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			if i-start >= 2 {
				out = append(out, string(r[start:i]))
			}
			start = -1
		}
	}
	if start >= 0 && len(r)-start >= 2 {
		out = append(out, string(r[start:]))
	}
	return out
}
func chars(s string) []string {
	out := []string{}
	for _, w := range strings.Fields(strings.ToLower(s)) {
		r := append([]rune{' '}, []rune(w)...)
		r = append(r, ' ')
		for n := 3; n <= 5; n++ {
			for i := 0; i+n <= len(r); i++ {
				out = append(out, string(r[i:i+n]))
			}
		}
	}
	return out
}
func vector(tokens []string, f Feature) map[int]float64 {
	counts := map[int]float64{}
	for _, t := range tokens {
		if i, ok := f.Vocabulary[t]; ok {
			counts[i]++
		}
	}
	n := 0.0
	for i, c := range counts {
		v := c * f.IDF[i]
		if f.Sublinear {
			v = (1 + math.Log(c)) * f.IDF[i]
		}
		counts[i] = v
		n += v * v
	}
	n = math.Sqrt(n)
	if n > 0 {
		for i, v := range counts {
			counts[i] = v / n
		}
	}
	return counts
}
func calibrate(c Calibration, score float64) float64 {
	if c.Method == "sigmoid" {
		z := c.Coefficient*score + c.Intercept
		if z >= 0 {
			return 1 / (1 + math.Exp(-z))
		}
		e := math.Exp(z)
		return e / (1 + e)
	}
	if len(c.X) == 0 {
		return math.NaN()
	}
	if score <= c.X[0] {
		return c.Y[0]
	}
	last := len(c.X) - 1
	if score >= c.X[last] {
		return c.Y[last]
	}
	i := sort.SearchFloat64s(c.X, score)
	x0, x1 := c.X[i-1], c.X[i]
	return c.Y[i-1] + (score-x0)*(c.Y[i]-c.Y[i-1])/(x1-x0)
}
func classify(b Bundle, text string) (map[string]any, error) {
	text = norm.NFC.String(strings.TrimSpace(text))
	if text == "" {
		return nil, errors.New("text required")
	}
	base := words(text)
	wt := append([]string{}, base...)
	for i := 0; i+1 < len(base); i++ {
		wt = append(wt, base[i]+" "+base[i+1])
	}
	wv := vector(wt, b.Word)
	cv := vector(chars(text), b.Char)
	score := b.Intercept[0]
	for i, v := range wv {
		score += b.Coef[0][i] * v
	}
	off := len(b.Word.Vocabulary)
	for i, v := range cv {
		score += b.Coef[0][off+i] * v
	}
	// sklearn decision scores are oriented to the second class in sorted order;
	// the exported calibration is fit on positive-oriented scores. Mirror Python's
	// positive_scores() so the same calibration input is used for any label order.
	if len(b.Labels) == 2 && b.Labels[0] == b.Positive {
		score = -score
	}
	p := calibrate(b.Calibration, score)
	if math.IsNaN(p) || p < 0 || p > 1 {
		return nil, errors.New("invalid calibration")
	}
	label := b.Labels[0]
	if label == b.Positive {
		label = b.Labels[1]
	}
	if p >= b.Policy.Threshold {
		label = b.Positive
	}
	return map[string]any{"label": label, b.Probability: p, "confidence": p, "review_recommended": p >= b.Policy.Review[0] && p <= b.Policy.Review[1], "model_version": b.Version, "task": b.Task}, nil
}
func main() {
	model := flag.String("model", "model.json", "")
	text := flag.String("text", "", "")
	inputJSON := flag.String("input-json", "", "")
	flag.Parse()
	b, e := load(*model)
	if e == nil {
		input := *text
		if (*text == "") == (*inputJSON == "") {
			e = errors.New("provide exactly one of --text or --input-json")
		} else if *inputJSON != "" {
			var values map[string]string
			if e = json.Unmarshal([]byte(*inputJSON), &values); e == nil {
				if len(values) != len(b.Input.Fields) {
					e = errors.New("input fields do not match artifact")
				} else {
					parts := make([]string, len(b.Input.Fields))
					for i, field := range b.Input.Fields {
						value, ok := values[field]
						if !ok || strings.TrimSpace(value) == "" {
							e = errors.New("input fields do not match artifact")
							break
						}
						if len(b.Input.Fields) == 1 {
							parts[i] = strings.TrimSpace(value)
						} else {
							parts[i] = "<" + strings.ToUpper(field) + ">\n" + strings.TrimSpace(value)
						}
					}
					input = strings.Join(parts, "\n\n")
				}
			}
		}
		var out map[string]any
		if e == nil {
			out, e = classify(b, input)
		}
		if e == nil {
			e = json.NewEncoder(os.Stdout).Encode(out)
		}
	}
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(2)
	}
}
