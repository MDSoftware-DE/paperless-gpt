package ocr

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMapWordsToSpans_WithGlyphSpans(t *testing.T) {
	words := []string{"Jet", "Tankstelle", "5964"}
	spans := []syntheticWordSpan{
		{X1: 10, X2: 24},
		{X1: 30, X2: 41},
		{X1: 45, X2: 54},
		{X1: 58, X2: 69},
		{X1: 74, X2: 80},
	}

	out := mapWordsToSpans(words, spans, 8, 80, 200, 4)
	assert.Len(t, out, len(words))
	for i := range out {
		assert.Greater(t, out[i].X2, out[i].X1)
		assert.False(t, math.IsNaN(out[i].X1))
		assert.False(t, math.IsNaN(out[i].X2))
		if i > 0 {
			assert.Greater(t, out[i].X1, out[i-1].X1)
		}
	}
}

func TestMapWordsToSpans_WithNoGlyphSpansFallsBackToUniformDistribution(t *testing.T) {
	words := []string{"Diesel", "*", "EUR", "31,06"}
	out := mapWordsToSpans(words, nil, 12, 120, 200, 4)

	assert.Len(t, out, len(words))
	assert.InDelta(t, 12, out[0].X1, 0.01)
	assert.InDelta(t, 132, out[len(out)-1].X2, 0.01)
	for i := 0; i < len(out)-1; i++ {
		assert.Less(t, out[i].X2, out[i+1].X2)
	}
}

func TestMapWordsToSpans_ExpandsEdgeSpansToLineBounds(t *testing.T) {
	words := []string{"Jet", "Tankstelle", "5964"}
	spans := []syntheticWordSpan{
		{X1: 40, X2: 65},
		{X1: 90, X2: 140},
		{X1: 170, X2: 225},
	}

	out := mapWordsToSpans(words, spans, 20, 200, 300, 4)

	assert.Len(t, out, len(words))
	assert.InDelta(t, 20, out[0].X1, 0.01)
	assert.InDelta(t, 220, out[len(out)-1].X2, 40)
}

func TestMapWordsToSpans_EnforcesWordGap(t *testing.T) {
	words := []string{"A", "B", "C", "D"}
	out := mapWordsToSpans(words, nil, 10, 80, 200, 2)

	assert.Len(t, out, len(words))
	for i := 0; i < len(out)-1; i++ {
		assert.GreaterOrEqual(t, out[i+1].X1-out[i].X2, 1.0)
	}
}

func TestWordSpanMappingUsableRequiresLengthMatch(t *testing.T) {
	words := []string{"A", "B", "C"}
	spans := []syntheticWordSpan{{X1: 1, X2: 10}, {X1: 14, X2: 20}}
	assert.False(t, wordSpanMappingUsable(spans, words, 0, 20, 4))
}

func TestWordSpanMappingUsableRejectsTinyCoverage(t *testing.T) {
	words := []string{"A", "B"}
	spans := []syntheticWordSpan{{X1: 1, X2: 2}, {X1: 10, X2: 11}}
	assert.False(t, wordSpanMappingUsable(spans, words, 0, 20, 4))
}
