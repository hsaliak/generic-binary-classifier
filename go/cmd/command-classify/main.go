package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
)

type Feature struct {
	Vocabulary map[string]int `json:"vocabulary"`
	IDF        []float64      `json:"idf"`
	Ngram      []int          `json:"ngram_range"`
}
type Bundle struct {
	Labels    []string    `json:"labels"`
	Word      Feature     `json:"word"`
	Char      Feature     `json:"char"`
	Coef      [][]float64 `json:"coefficients"`
	Intercept []float64   `json:"intercept"`
}

func vector(text string, f Feature, chars bool) map[int]float64 {
	counts := map[int]float64{}
	tokens := []string{}
	if chars {
		for _, w := range regexp.MustCompile(`\S+`).FindAllString(text, -1) {
			p := " " + w + " "
			for n := f.Ngram[0]; n <= f.Ngram[1]; n++ {
				for i := 0; i+n <= len(p); i++ {
					tokens = append(tokens, p[i:i+n])
				}
			}
		}
	} else {
		tokens = regexp.MustCompile(`(?i)[[:alnum:]_]{2,}`).FindAllString(text, -1)
		base := append([]string{}, tokens...)
		for i := 0; i+1 < len(base); i++ {
			tokens = append(tokens, base[i]+" "+base[i+1])
		}
	}
	for _, t := range tokens {
		if i, ok := f.Vocabulary[t]; ok {
			counts[i]++
		}
	}
	norm := 0.0
	for i, c := range counts {
		v := (1 + math.Log(c)) * f.IDF[i]
		counts[i] = v
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i, v := range counts {
			counts[i] = v / norm
		}
	}
	return counts
}
func main() {
	model := flag.String("model", "model.json", "")
	text := flag.String("text", "", "")
	flag.Parse()
	raw, e := os.ReadFile(*model)
	if e != nil {
		panic(e)
	}
	var b Bundle
	if e = json.Unmarshal(raw, &b); e != nil {
		panic(e)
	}
	score := b.Intercept[0]
	offset := len(b.Word.Vocabulary)
	for i, v := range vector(*text, b.Word, false) {
		score += b.Coef[0][i] * v
	}
	for i, v := range vector(*text, b.Char, true) {
		score += b.Coef[0][offset+i] * v
	}
	p := 1 / (1 + math.Exp(-score))
	label := "safe"
	if p >= .5 {
		label = "unsafe"
	}
	out := map[string]any{"label": label, "unsafe_probability": p, "review_recommended": p >= .2 && p <= .9}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if strings.TrimSpace(*text) == "" {
		panic("text required")
	}
	fmt.Print("")
	enc.Encode(out)
}
