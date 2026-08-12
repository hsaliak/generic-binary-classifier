package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

func TestClassifyUsesPolicyAndCalibration(t *testing.T) {
	b := Bundle{Format: 3, Version: "v", Labels: []string{"safe", "unsafe"}, Positive: "unsafe", Probability: "positive_probability", Input: Input{Fields: []string{"text"}}, Word: Feature{Vocabulary: map[string]int{"status": 0}, IDF: []float64{1}, Ngram: []int{1, 2}, Sublinear: true, Norm: "l2"}, Char: Feature{Vocabulary: map[string]int{}, IDF: []float64{}, Ngram: []int{3, 5}, Sublinear: true, Norm: "l2"}, Coef: [][]float64{{1}}, Intercept: []float64{0}, Calibration: Calibration{Method: "sigmoid", Coefficient: 1}, Policy: Policy{Threshold: .5, Review: []float64{.2, .9}}}
	out, err := classify(b, "STATUS")
	if err != nil || out["label"] != "unsafe" {
		t.Fatalf("classify: %v %#v", err, out)
	}
}

// The exported calibration is fit on positive-oriented scores, and sklearn orients
// raw decision scores to the second class in sorted order. When the positive class
// sorts first, the score must be negated before calibration, exactly as Python does.
func TestClassifyNegatesScoreWhenPositiveSortsFirst(t *testing.T) {
	b := Bundle{Format: 3, Version: "v", Labels: []string{"high_complexity", "low_complexity"}, Positive: "high_complexity", Probability: "positive_probability", Input: Input{Fields: []string{"text"}}, Word: Feature{Vocabulary: map[string]int{"status": 0}, IDF: []float64{1}, Ngram: []int{1, 2}, Sublinear: true, Norm: "l2"}, Char: Feature{Vocabulary: map[string]int{}, IDF: []float64{}, Ngram: []int{3, 5}, Sublinear: true, Norm: "l2"}, Coef: [][]float64{{5}}, Intercept: []float64{0}, Calibration: Calibration{Method: "sigmoid", Coefficient: 5}, Policy: Policy{Threshold: .5, Review: []float64{.2, .9}}}
	out, err := classify(b, "STATUS")
	if err != nil {
		t.Fatal(err)
	}
	// Raw score +5 would give sigmoid(25) if un-negated; positive-first ordering
	// negates to -5, so the calibrated probability must be small and the label
	// must fall back to the negative class at the 0.5 threshold.
	if out["label"] != "low_complexity" {
		t.Fatalf("expected negative label after score negation: %#v", out)
	}
	if p := out["positive_probability"].(float64); p >= 0.01 {
		t.Fatalf("expected negated score to yield small positive probability, got %v", p)
	}
}

func FuzzV3ArtifactParsing(f *testing.F) {
	f.Add([]byte(`{"format_version":3}`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		path := t.TempDir() + "/model.json"
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(raw)
		if err := os.WriteFile(path+".sha256", []byte(hex.EncodeToString(sum[:])+"  model.json\n"), 0600); err != nil {
			t.Fatal(err)
		}
		_, _ = load(path)
	})
}

func FuzzClassifyNeverReturnsInvalidProbability(f *testing.F) {
	f.Add("git status")
	f.Fuzz(func(t *testing.T, text string) {
		b := Bundle{Format: 3, Version: "v", Labels: []string{"safe", "unsafe"}, Positive: "unsafe", Probability: "positive_probability", Input: Input{Fields: []string{"text"}}, Word: Feature{Vocabulary: map[string]int{}, IDF: []float64{}, Ngram: []int{1, 2}, Sublinear: true, Norm: "l2"}, Char: Feature{Vocabulary: map[string]int{}, IDF: []float64{}, Ngram: []int{3, 5}, Sublinear: true, Norm: "l2"}, Coef: [][]float64{{}}, Intercept: []float64{0}, Calibration: Calibration{Method: "sigmoid", Coefficient: 1}, Policy: Policy{Threshold: .5, Review: []float64{.2, .9}}}
		out, err := classify(b, text)
		if strings.TrimSpace(text) == "" {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		p := out["positive_probability"].(float64)
		if p < 0 || p > 1 {
			t.Fatal(p)
		}
	})
}
