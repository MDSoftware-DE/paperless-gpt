package ocr

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	_ "image/jpeg"

	"github.com/gardar/ocrchestra/pkg/hocr"
	"github.com/sirupsen/logrus"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
	"github.com/tmc/langchaingo/llms/mistral"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
)

// LLMProvider implements OCR using LLM vision models
type LLMProvider struct {
	provider    string
	model       string
	llm         llms.Model
	prompt      string
	maxTokens   int
	temperature *float64
	ollamaTopK  *int

	// hOCR state for downstream PDF text-layer generation.
	enableHOCR bool
	mu         sync.Mutex
	hocrPages  []hocr.Page
}

type syntheticLayoutRegion struct {
	X1    float64
	Y1    float64
	X2    float64
	Y2    float64
	Valid bool
}

type syntheticLineBand struct {
	X1        float64
	Y1        float64
	X2        float64
	Y2        float64
	WordSpans []syntheticWordSpan
}

type syntheticWordSpan struct {
	X1 float64
	X2 float64
}

func newLLMProvider(config Config) (*LLMProvider, error) {
	logger := log.WithFields(logrus.Fields{
		"provider": config.VisionLLMProvider,
		"model":    config.VisionLLMModel,
	})
	logger.Info("Creating new LLM OCR provider")

	var model llms.Model
	var err error

	switch strings.ToLower(config.VisionLLMProvider) {
	case "openai":
		logger.Debug("Initializing OpenAI vision model")
		model, err = createOpenAIClient(config)
	case "ollama":
		logger.Debug("Initializing Ollama vision model")
		model, err = createOllamaClient(config)
	case "mistral":
		logger.Debug("Initializing Mistral vision model")
		model, err = createMistralClient(config)
	case "anthropic":
		logger.Debug("Initializing Anthropic vision model")
		model, err = createAnthropicClient(config)
	default:
		return nil, fmt.Errorf("unsupported vision LLM provider: %s", config.VisionLLMProvider)
	}

	if err != nil {
		logger.WithError(err).Error("Failed to create vision LLM client")
		return nil, fmt.Errorf("error creating vision LLM client: %w", err)
	}

	logger.Info("Successfully initialized LLM OCR provider")
	return &LLMProvider{
		provider:    config.VisionLLMProvider,
		model:       config.VisionLLMModel,
		llm:         model,
		prompt:      config.VisionLLMPrompt,
		maxTokens:   config.VisionLLMMaxTokens,
		temperature: config.VisionLLMTemperature,
		ollamaTopK:  config.OllamaOcrTopK,
		enableHOCR:  config.EnableHOCR,
		hocrPages:   make([]hocr.Page, 0),
	}, nil
}

func (p *LLMProvider) ProcessImage(ctx context.Context, imageContent []byte, pageNumber int) (*OCRResult, error) {
	// Attach per-request metadata for downstream routers / artifact storage.
	ctx = WithRequestMeta(ctx, RequestMeta{PageNumber: pageNumber})

	logger := log.WithFields(logrus.Fields{
		"provider": p.provider,
		"model":    p.model,
		"page":     pageNumber,
	})
	logger.Debug("Starting LLM OCR processing")

	// Log the image dimensions
	img, _, err := image.Decode(bytes.NewReader(imageContent))
	if err != nil {
		logger.WithError(err).Error("Failed to decode image")
		return nil, fmt.Errorf("error decoding image: %w", err)
	}
	bounds := img.Bounds()
	layoutRegion := detectTextRegion(img)
	lineBands := detectTextLineBands(img, layoutRegion)
	logger.WithFields(logrus.Fields{
		"width":               bounds.Dx(),
		"height":              bounds.Dy(),
		"layout_region_valid": layoutRegion.Valid,
		"line_bands":          len(lineBands),
	}).Debug("Image dimensions")

	logger.Debugf("Prompt: %s", p.prompt)

	// Prepare content parts based on provider type
	var parts []llms.ContentPart
	var imagePart llms.ContentPart
	providerName := strings.ToLower(p.provider)

	if providerName == "openai" || providerName == "mistral" {
		logger.Info("Using OpenAI image format")
		imagePart = llms.ImageURLPart("data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(imageContent))
	} else {
		logger.Info("Using binary image format")
		imagePart = llms.BinaryPart("image/jpeg", imageContent)
	}

	parts = []llms.ContentPart{
		imagePart,
		llms.TextPart(p.prompt),
	}

	var callOpts []llms.CallOption
	if p.maxTokens > 0 {
		callOpts = append(callOpts, llms.WithMaxTokens(p.maxTokens))
	}
	if p.temperature != nil {
		callOpts = append(callOpts, llms.WithTemperature(*p.temperature))
	}
	if providerName == "ollama" && p.ollamaTopK != nil {
		callOpts = append(callOpts, llms.WithTopK(*p.ollamaTopK))
	}

	// Convert the image to text
	logger.Debug("Sending request to vision model")
	completion, err := p.llm.GenerateContent(ctx, []llms.MessageContent{
		{
			Parts: parts,
			Role:  llms.ChatMessageTypeHuman,
		},
	}, callOpts...)
	if err != nil {
		logger.WithError(err).Error("Failed to get response from vision model")
		return nil, fmt.Errorf("error getting response from LLM: %w", err)
	}

	text := stripReasoning(completion.Choices[0].Content)
	limitHit := false
	tokenCount := -1

	if p.maxTokens > 0 {
		genInfo := completion.Choices[0].GenerationInfo
		if genInfo != nil && genInfo["TotalTokens"] != nil {
			if v, ok := genInfo["TotalTokens"].(int); ok {
				tokenCount = v
			}
		}
		// Fallback: count tokens using langchaingo (might not be accurate for all models)
		if tokenCount < 0 {
			tokenCount = llms.CountTokens(p.model, text)
		}
		if tokenCount >= p.maxTokens {
			limitHit = true
		}
	}

	result := &OCRResult{
		Text: text,
		Metadata: map[string]string{
			"provider": p.provider,
			"model":    p.model,
		},
		OcrLimitHit:    limitHit,
		GenerationInfo: completion.Choices[0].GenerationInfo,
	}

	// The LLM OCR providers do not return layout coordinates.
	// Build a synthetic hOCR page so downstream PDF generation can still
	// create a selectable text layer for archive consistency.
	if p.enableHOCR {
		hocrPage := buildSyntheticHOCRPage(text, bounds.Dx(), bounds.Dy(), pageNumber, p.model, layoutRegion, lineBands)
		p.mu.Lock()
		p.hocrPages = append(p.hocrPages, hocrPage)
		p.mu.Unlock()
		result.HOCRPage = &hocrPage
	}

	logger.WithField("content_length", len(result.Text)).WithFields(completion.Choices[0].GenerationInfo).Info("Successfully processed image")
	return result, nil
}

// IsHOCREnabled returns whether synthetic hOCR generation is enabled.
func (p *LLMProvider) IsHOCREnabled() bool {
	return p.enableHOCR
}

// GetHOCRPages returns a copy of collected synthetic hOCR pages.
func (p *LLMProvider) GetHOCRPages() []hocr.Page {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]hocr.Page, len(p.hocrPages))
	copy(out, p.hocrPages)
	return out
}

// GetHOCRDocument returns a synthetic hOCR document for all processed pages.
func (p *LLMProvider) GetHOCRDocument() (*hocr.HOCR, error) {
	if !p.enableHOCR {
		return nil, fmt.Errorf("hOCR generation is not enabled")
	}

	p.mu.Lock()
	if len(p.hocrPages) == 0 {
		p.mu.Unlock()
		return nil, fmt.Errorf("no hOCR pages collected")
	}
	pages := make([]hocr.Page, len(p.hocrPages))
	copy(pages, p.hocrPages)
	p.mu.Unlock()

	sort.Slice(pages, func(i, j int) bool {
		return pages[i].PageNumber < pages[j].PageNumber
	})

	doc := &hocr.HOCR{
		Title:    "LLM OCR Document",
		Language: "unknown",
		Metadata: map[string]string{
			"ocr-system":          "paperless-gpt-llm-synthetic-hocr",
			"ocr-number-of-pages": fmt.Sprintf("%d", len(pages)),
		},
		Pages: pages,
	}
	return doc, nil
}

// ResetHOCR clears stored synthetic hOCR pages.
func (p *LLMProvider) ResetHOCR() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.hocrPages = make([]hocr.Page, 0)
}

func buildSyntheticHOCRPage(
	text string,
	width, height, pageNumber int,
	model string,
	layoutRegion syntheticLayoutRegion,
	lineBands []syntheticLineBand,
) hocr.Page {
	w := float64(width)
	h := float64(height)
	if w <= 0 {
		w = 2480
	}
	if h <= 0 {
		h = 3508
	}
	originalW := w
	originalH := h
	w, h = normalizeSyntheticPageSize(w, h)
	scaleX := 1.0
	scaleY := 1.0
	if originalW > 0 {
		scaleX = w / originalW
	}
	if originalH > 0 {
		scaleY = h / originalH
	}

	lines := make([]string, 0)
	emptyStreak := 0
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			if len(lines) > 0 && emptyStreak == 0 {
				lines = append(lines, "")
			}
			emptyStreak++
			continue
		}
		emptyStreak = 0
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		fallback := strings.TrimSpace(text)
		if fallback != "" {
			lines = append(lines, fallback)
		}
	}
	if len(lines) == 0 {
		lines = append(lines, " ")
	}

	nonEmptyLines := 0
	maxLineRunes := 1
	layoutRows := 0.0
	for _, line := range lines {
		if line == "" {
			layoutRows += 0.6
			continue
		}
		nonEmptyLines++
		maxLineRunes = maxInt(maxLineRunes, utf8.RuneCountInString(line))
		layoutRows += 1.0
	}
	if nonEmptyLines == 0 {
		nonEmptyLines = 1
	}
	if layoutRows <= 0 {
		layoutRows = 1.0
	}

	marginX := maxFloat(8, w*0.018)
	topMargin := maxFloat(6, h*0.010)
	bottomMargin := maxFloat(6, h*0.010)
	usableWidth := maxFloat(10, w-(2*marginX))
	usableHeight := maxFloat(10, h-topMargin-bottomMargin)
	if layoutRegion.Valid {
		regionX1 := clampFloat(layoutRegion.X1*scaleX, 0, w-10)
		regionY1 := clampFloat(layoutRegion.Y1*scaleY, 0, h-10)
		regionX2 := clampFloat(layoutRegion.X2*scaleX, regionX1+5, w)
		regionY2 := clampFloat(layoutRegion.Y2*scaleY, regionY1+5, h)
		regionWidth := regionX2 - regionX1
		regionHeight := regionY2 - regionY1

		// Horizontal bounds are often reliable even when vertical bounds are noisy.
		if regionWidth > w*0.15 {
			marginX = clampFloat(regionX1-maxFloat(2, w*0.004), 2, w-15)
			usableWidth = clampFloat(
				regionWidth+maxFloat(4, w*0.008),
				20,
				maxFloat(20, w-marginX-5),
			)
		}
		// Only trust vertical bounds when they cover a substantial share of the page.
		if regionHeight > h*0.55 {
			topMargin = clampFloat(regionY1-maxFloat(4, h*0.010), 2, h-15)
			usableHeight = clampFloat(
				regionHeight+maxFloat(6, h*0.014),
				20,
				maxFloat(20, h-topMargin-5),
			)
			bottomMargin = maxFloat(5, h-(topMargin+usableHeight))
		}
	}
	lineAdvance := clampFloat(usableHeight/layoutRows, 4.8, 18)
	lineBoxHeight := clampFloat(lineAdvance*0.58, 2.6, lineAdvance*0.74)
	yCursor := topMargin

	scaledBands := make([]syntheticLineBand, 0, len(lineBands))
	for _, b := range lineBands {
		sb := syntheticLineBand{
			X1: clampFloat(b.X1*scaleX, 0, w-2),
			Y1: clampFloat(b.Y1*scaleY, 0, h-2),
			X2: clampFloat(b.X2*scaleX, 0, w-2),
			Y2: clampFloat(b.Y2*scaleY, 0, h-2),
		}
		if len(b.WordSpans) > 0 {
			sb.WordSpans = make([]syntheticWordSpan, 0, len(b.WordSpans))
			for _, ws := range b.WordSpans {
				x1 := clampFloat(ws.X1*scaleX, 0, w-2)
				x2 := clampFloat(ws.X2*scaleX, x1+1, w-2)
				if x2 <= x1+0.8 {
					continue
				}
				sb.WordSpans = append(sb.WordSpans, syntheticWordSpan{X1: x1, X2: x2})
			}
		}
		if sb.X2 <= sb.X1+2 || sb.Y2 <= sb.Y1+1 {
			continue
		}
		// Filter out tiny/noisy bands; keep only rows that plausibly contain full text lines.
		if (sb.X2 - sb.X1) < maxFloat(usableWidth*0.28, 35) {
			continue
		}
		scaledBands = append(scaledBands, sb)
	}
	bandRatio := 0.0
	if nonEmptyLines > 0 {
		bandRatio = float64(len(scaledBands)) / float64(nonEmptyLines)
	}
	minBandsRequired := minInt(nonEmptyLines, 6)
	if minBandsRequired < 3 {
		minBandsRequired = 3
	}
	// Accept a broader ratio window: line-band detection is often slightly over/under
	// segmented, but still good enough to anchor word-level x positions.
	useBands := len(scaledBands) >= minBandsRequired && bandRatio >= 0.70 && bandRatio <= 1.35

	// Strict but non-destructive vertical text corridor.
	// Keep bottom close to the synthetic layout area to avoid clipping long documents.
	textTop := clampFloat(topMargin, 2, h-20)
	textBottom := clampFloat(topMargin+usableHeight, textTop+20, h-2)

	renderedLines := 0
	lastY2 := textTop - lineBoxHeight
	// Start directly at the detected/derived top so early lines are not lost.
	yCursor = textTop

	hocrLines := make([]hocr.Line, 0, len(lines))
	for i, lineText := range lines {
		if lineText == "" {
			yCursor += lineAdvance * 0.35
			continue
		}

		remainingLines := maxInt(nonEmptyLines-renderedLines, 1)
		availableHeight := maxFloat(6, textBottom-yCursor)
		dynAdvance := minFloat(lineAdvance, availableHeight/float64(remainingLines))
		dynAdvance = clampFloat(dynAdvance, 2.8, lineAdvance)
		dynBoxHeight := clampFloat(minFloat(lineBoxHeight, dynAdvance*0.72), 2.4, dynAdvance*0.72)

		fallbackY1 := yCursor
		y1 := fallbackY1
		if y1 >= textBottom-2.2 {
			break
		}
		fallbackY2 := minFloat(y1+dynBoxHeight, textBottom)
		y2 := fallbackY2
		yCursor += dynAdvance

		words := normalizeSyntheticWords(strings.Fields(lineText))
		if len(words) == 0 {
			words = []string{" "}
		}

		lineRuneCount := 0
		for _, word := range words {
			lineRuneCount += maxInt(utf8.RuneCountInString(word), 1)
		}
		lineSpaceCount := maxInt(len(words)-1, 0)
		if lineRuneCount+lineSpaceCount <= 0 {
			lineRuneCount = 1
		}
		lineMarginX := marginX
		lineUsableWidth := usableWidth
		lineUnits := float64(lineRuneCount + lineSpaceCount)
		baseCharWidth := clampFloat(usableWidth/float64(maxLineRunes), 2.2, 5.2)
		lineContentWidth := estimateLineTextWidth(words, baseCharWidth)
		lineContentWidth = clampFloat(lineContentWidth, 12.0, lineUsableWidth)
		lineCharWidth := clampFloat(lineContentWidth/lineUnits, 2.2, 5.0)
		lineContentWidth = clampFloat(lineContentWidth, lineCharWidth*lineUnits, lineUsableWidth)
		lineSpaceWidth := maxFloat(lineCharWidth*1.12, 2.0)
		mappedWordSpans := []syntheticWordSpan(nil)
		if useBands {
			bi := mapLineIndex(renderedLines, nonEmptyLines, len(scaledBands))
			if bi >= 0 && bi < len(scaledBands) {
				band := scaledBands[bi]
				// Keep X placement stable from the synthetic model to avoid horizontal jitter.
				// Use detected bands only as a bounded vertical hint.
				fallbackCenter := (fallbackY1 + fallbackY2) / 2
				bandCenter := (band.Y1 + band.Y2) / 2
				maxShift := maxFloat(1.8, dynAdvance*0.22)
				centerShift := clampFloat(bandCenter-fallbackCenter, -maxShift, maxShift)
				targetCenter := fallbackCenter + centerShift
				targetHeight := minFloat(band.Y2-band.Y1, dynAdvance*0.72)
				targetHeight = clampFloat(targetHeight, 2.4, dynAdvance*0.72)
				y1 = targetCenter - (targetHeight / 2)
				y2 = targetCenter + (targetHeight / 2)

				// For horizontal placement, trust the detected band bounds with a strong blend.
				// This keeps right-edge alignment close to the visible text on scans/receipts.
				detectedX1 := clampFloat(band.X1-maxFloat(2, w*0.004), 2, w-20)
				detectedX2 := clampFloat(band.X2+maxFloat(4, w*0.010), detectedX1+20, w-2)
				detectedWidth := detectedX2 - detectedX1
				if detectedWidth >= maxFloat(20, usableWidth*0.20) {
					baseRight := lineMarginX + lineUsableWidth
					blendX := 0.70
					lineMarginX = clampFloat((lineMarginX*(1.0-blendX))+(detectedX1*blendX), 2, w-20)
					lineRight := clampFloat((baseRight*(1.0-blendX))+(detectedX2*blendX), lineMarginX+20, w-2)
					lineUsableWidth = maxFloat(20, lineRight-lineMarginX)
					lineContentWidth = clampFloat(lineRight-lineMarginX, lineUsableWidth*0.16, lineUsableWidth)
					lineCharWidth = clampFloat(lineContentWidth/lineUnits, 2.2, 5.0)
					lineSpaceWidth = maxFloat(lineCharWidth*1.12, 2.0)
				}

				mappedWordSpans = mapWordsToSpans(
					words,
					band.WordSpans,
					lineMarginX,
					lineContentWidth,
					w,
					lineCharWidth,
				)
				if !wordSpanMappingUsable(
					mappedWordSpans,
					words,
					lineMarginX,
					lineContentWidth,
					lineCharWidth,
				) {
					mappedWordSpans = nil
				}
			}
		}

		// Enforce monotonic line progression; overlapping lines make UI text selection unreliable.
		minY1 := lastY2 + maxFloat(1.4, dynAdvance*0.75)
		if y1 < minY1 {
			shift := minY1 - y1
			y1 += shift
			y2 += shift
		}
		if y2 > textBottom {
			shift := y2 - textBottom
			y1 -= shift
			y2 -= shift
		}
		if y1 < textTop {
			shift := textTop - y1
			y1 += shift
			y2 += shift
		}

		y1 = clampFloat(y1, textTop, textBottom-2.2)
		y2 = clampFloat(maxFloat(y2, y1+2.2), y1+2.2, textBottom)

		// Prevent accidental bleed into neighboring lines when selection is rendered by PDF viewers.
		lineMaxHeight := dynAdvance * 0.4
		if y2-y1 > lineMaxHeight {
			y2 = y1 + lineMaxHeight
		}
		if y2 > textBottom {
			y2 = textBottom
		}

		// Keep the final rendered line strictly inside the visible text corridor.
		if remainingLines == 1 {
			hardBottom := textBottom - maxFloat(0.6, dynBoxHeight*0.16)
			if hardBottom > y1+2.2 {
				y2 = minFloat(y2, hardBottom)
			}
		}
		lastY2 = y2

		fallbackLineSpaceX := lineMarginX
		fallbackLineEnd := lineMarginX + lineContentWidth
		if len(mappedWordSpans) == 0 {
			fallbackLineSpaceX = lineMarginX
			fallbackLineEnd = minFloat(lineMarginX+lineUsableWidth, w-2)
		} else if len(mappedWordSpans) == len(words) {
			fallbackLineSpaceX = mappedWordSpans[0].X1
			fallbackLineEnd = mappedWordSpans[len(mappedWordSpans)-1].X2
		}

		wordObjs := make([]hocr.Word, 0, len(words))
		for wIdx, word := range words {
			charCount := maxInt(utf8.RuneCountInString(word), 1)
			var x1, x2 float64
			if wIdx < len(mappedWordSpans) {
				x1 = mappedWordSpans[wIdx].X1
				x2 = mappedWordSpans[wIdx].X2
			} else {
				ww := maxFloat(float64(charCount)*lineCharWidth, lineCharWidth)
				x1 = fallbackLineSpaceX
				x2 = minFloat(x1+ww, fallbackLineEnd)
				fallbackLineSpaceX = minFloat(x2+lineSpaceWidth, fallbackLineEnd)
			}
			if x1 >= w-2 {
				break
			}
			if x2 <= x1 {
				x2 = minFloat(x1+6, w-2)
			}
			wordObjs = append(wordObjs, hocr.Word{
				ID:         fmt.Sprintf("word_%d_%d_%d", pageNumber, i+1, wIdx+1),
				Text:       word,
				BBox:       hocr.NewBoundingBox(x1, y1, x2, y2),
				Confidence: 85,
				Lang:       "unknown",
			})
		}

		lineLeft := fallbackLineSpaceX
		lineRight := minFloat(lineMarginX+lineContentWidth, w-2)
		if len(wordObjs) > 0 {
			lineLeft = wordObjs[0].BBox.X1
			lineRight = wordObjs[len(wordObjs)-1].BBox.X2
		} else if len(mappedWordSpans) > 0 {
			lineLeft = mappedWordSpans[0].X1
			lineRight = mappedWordSpans[len(mappedWordSpans)-1].X2
		}
		lineBBox := hocr.NewBoundingBox(lineLeft, y1, lineRight, y2)
		if len(wordObjs) > 0 {
			lineBBox = hocr.NewBoundingBox(
				wordObjs[0].BBox.X1,
				y1,
				wordObjs[len(wordObjs)-1].BBox.X2,
				y2,
			)
		}

		hocrLines = append(hocrLines, hocr.Line{
			ID:       fmt.Sprintf("line_%d_%d", pageNumber, i+1),
			Lang:     "unknown",
			BBox:     lineBBox,
			Baseline: "0 0",
			Words:    wordObjs,
		})
		renderedLines++
	}

	pageMeta := map[string]string{
		"synthetic": "true",
		"model":     model,
	}
	if layoutRegion.Valid {
		pageMeta["layout_region"] = fmt.Sprintf("%.1f,%.1f,%.1f,%.1f", layoutRegion.X1, layoutRegion.Y1, layoutRegion.X2, layoutRegion.Y2)
	}

	return hocr.Page{
		ID:         fmt.Sprintf("page_%d", pageNumber),
		Title:      fmt.Sprintf("synthetic_hocr model=%s", model),
		PageNumber: pageNumber,
		ImageName:  fmt.Sprintf("page%03d.jpg", pageNumber),
		Lang:       "unknown",
		BBox:       hocr.NewBoundingBox(0, 0, w, h),
		Lines:      hocrLines,
		Metadata:   pageMeta,
	}
}

func mapLineIndex(lineIdx, totalLines, totalBands int) int {
	if totalLines <= 0 || totalBands <= 0 {
		return 0
	}
	if totalLines == 1 {
		return 0
	}
	if lineIdx < 0 {
		lineIdx = 0
	}
	if lineIdx >= totalLines {
		lineIdx = totalLines - 1
	}
	pos := float64(lineIdx) / float64(totalLines-1)
	idx := int(pos * float64(totalBands-1))
	return clampInt(idx, 0, totalBands-1)
}

func mapWordsToSpans(
	words []string,
	spans []syntheticWordSpan,
	lineX, lineWidth, pageW float64,
	estimatedCharWidth float64,
) []syntheticWordSpan {
	if len(words) == 0 {
		return nil
	}
	lineStart := clampFloat(lineX, 0, pageW-2)
	lineEnd := clampFloat(lineX+lineWidth, lineStart+2, pageW-1)
	lineStart = clampFloat(lineStart, 0, lineEnd-1)
	lineEnd = clampFloat(lineEnd, lineStart+2, pageW-1)

	clean := make([]syntheticWordSpan, 0, len(spans))
	for _, s := range spans {
		x1 := clampFloat(s.X1, lineStart, lineEnd-1)
		x2 := clampFloat(s.X2, x1+1, lineEnd)
		if x2-x1 < 0.8 {
			continue
		}
		clean = append(clean, syntheticWordSpan{X1: x1, X2: x2})
	}

	base := distributeWordSpansByText(words, lineStart, lineEnd)
	if len(base) == 0 {
		return nil
	}

	out := base
	if len(clean) > 0 {
		if warped := warpWordSpansToGlyphSpans(base, clean, estimatedCharWidth); len(warped) == len(base) {
			out = warped
		}
	}

	// Enforce strict gutters and clamp to line bounds.
	// This makes selections easier to split to word level in PDF text overlays.
	minGap := clampFloat(maxFloat(estimatedCharWidth*0.22, 1.0), 1.0, 3.2)
	for i := 0; i < len(out)-1; i++ {
		if out[i].X2+minGap > out[i+1].X1 {
			mid := (out[i].X2 + out[i+1].X1) / 2
			out[i].X2 = mid - (minGap / 2)
			out[i+1].X1 = mid + (minGap / 2)
		}
	}

	for i := range out {
		out[i].X1 = clampFloat(out[i].X1, lineStart, lineEnd-0.9)
		out[i].X2 = clampFloat(maxFloat(out[i].X2, out[i].X1+1.2), out[i].X1+1.2, lineEnd)
	}
	// Keep widths readable and avoid huge width amplification caused by noisy spans.
	minSpanWidth := maxFloat(2.0, estimatedCharWidth*0.8)
	for i := range out {
		if out[i].X2-out[i].X1 < minSpanWidth {
			out[i].X2 = minFloat(out[i].X1+minSpanWidth, lineEnd)
		}
	}

	// Prefer first/last spans to occupy available horizontal area so the visible
	// line edges can be selected consistently.
	if len(out) > 0 {
		if out[0].X1 > lineStart+minSpanWidth {
			out[0].X1 = lineStart
		}
		if out[len(out)-1].X2 < lineEnd-minSpanWidth {
			out[len(out)-1].X2 = lineEnd
		}
	}
	return out
}

func warpWordSpansToGlyphSpans(
	base []syntheticWordSpan,
	spans []syntheticWordSpan,
	estimatedCharWidth float64,
) []syntheticWordSpan {
	if len(base) == 0 || len(spans) == 0 {
		return base
	}
	if len(spans) == 1 {
		span := spans[0]
		for i := range base {
			base[i] = span
		}
		return base
	}

	coverageTotal := 0.0
	spanPrefix := make([]float64, len(spans))
	for i, span := range spans {
		w := maxFloat(span.X2-span.X1, 0)
		coverageTotal += w
		spanPrefix[i] = coverageTotal
	}
	if coverageTotal <= 0 {
		return base
	}

	coverageToX := func(cov float64) float64 {
		if cov <= 0 {
			return spans[0].X1
		}
		if cov >= coverageTotal {
			return spans[len(spans)-1].X2
		}
		remaining := cov
		for i, span := range spans {
			prev := 0.0
			if i > 0 {
				prev = spanPrefix[i-1]
			}
			spanW := maxFloat(spanPrefix[i]-prev, 0)
			if remaining > spanW {
				remaining -= spanW
				continue
			}
			pos := span.X1 + remaining
			if pos > span.X2 {
				pos = span.X2
			}
			return pos
		}
		return spans[len(spans)-1].X2
	}

	lineStart := base[0].X1
	lineEnd := base[len(base)-1].X2
	lineWidth := maxFloat(lineEnd-lineStart, 1)
	minSpanWidth := clampFloat(estimatedCharWidth*0.55, 1.2, 3.4)

	// Build text boundaries and remap them to glyph coverage.
	remapped := make([]float64, 0, len(base)+1)
	remapped = append(remapped, base[0].X1)
	for _, w := range base {
		remapped = append(remapped, w.X2)
	}
	for i := 0; i < len(remapped); i++ {
		ratio := clampFloat((remapped[i]-lineStart)/lineWidth, 0, 1)
		remapped[i] = coverageToX(ratio * coverageTotal)
	}

	out := make([]syntheticWordSpan, 0, len(base))
	for i := 0; i < len(base); i++ {
		x1 := remapped[i]
		x2 := remapped[i+1]
		if x1 > x2 {
			x1, x2 = x2, x1
		}
		if x2-x1 < minSpanWidth {
			center := (x1 + x2) / 2
			x1 = center - (minSpanWidth / 2)
			x2 = center + (minSpanWidth / 2)
		}
		out = append(out, syntheticWordSpan{
			X1: x1,
			X2: x2,
		})
	}
	return out
}

func estimateLineTextWidth(words []string, estimatedCharWidth float64) float64 {
	if len(words) == 0 {
		return 0
	}
	if estimatedCharWidth <= 0 {
		estimatedCharWidth = 3.8
	}

	estimatedCharWidth = clampFloat(estimatedCharWidth, 1.6, 10.0)

	widthUnits := 0.0
	for _, word := range words {
		wordLen := float64(maxInt(utf8.RuneCountInString(word), 1))
		if isStandaloneSyntheticPunct(word) {
			wordLen *= 0.65
		}
		widthUnits += wordLen
	}
	if len(words) > 1 {
		widthUnits += float64(len(words)-1) * 0.9
	}
	if widthUnits <= 0 {
		widthUnits = 1
	}
	return widthUnits * estimatedCharWidth
}

func wordSpanMappingUsable(
	spans []syntheticWordSpan,
	words []string,
	lineX, lineWidth float64,
	estimatedCharWidth float64,
) bool {
	if len(words) == 0 || len(spans) != len(words) {
		return false
	}
	if lineWidth <= 2 {
		return false
	}
	if estimatedCharWidth <= 0 {
		return false
	}

	lineStart := lineX
	lineEnd := lineX + lineWidth
	prevX2 := lineStart - 1
	totalSpanWidth := 0.0
	nonTiny := 0
	for i, s := range spans {
		if s.X2 <= s.X1+0.6 {
			return false
		}
		spanWidth := s.X2 - s.X1
		word := words[i]
		wordChars := maxInt(utf8.RuneCountInString(word), 1)
		estimatedMin := maxFloat(estimatedCharWidth*0.45*float64(wordChars), 1.8)
		estimatedMax := maxFloat(estimatedCharWidth*4.0*float64(wordChars), 14.0)
		if spanWidth < estimatedMin || spanWidth > estimatedMax {
			return false
		}
		if i > 0 && s.X1 < prevX2-0.8 {
			return false
		}
		totalSpanWidth += spanWidth
		if spanWidth >= 1.2 {
			nonTiny++
		}
		prevX2 = s.X2
	}
	if nonTiny < maxInt(1, len(words)/2) {
		return false
	}

	coverage := spans[len(spans)-1].X2 - spans[0].X1
	if coverage < lineWidth*0.12 {
		return false
	}
	if totalSpanWidth > lineWidth*1.9 {
		return false
	}
	if spans[0].X1 > lineEnd || spans[len(spans)-1].X2 < lineStart {
		return false
	}
	return true
}

func distributeWordSpansByText(words []string, x1, x2 float64) []syntheticWordSpan {
	if len(words) == 0 {
		return nil
	}
	if x2 <= x1+1 {
		out := make([]syntheticWordSpan, len(words))
		step := 1.0
		for i := range words {
			out[i] = syntheticWordSpan{X1: x1 + (float64(i) * step), X2: x1 + (float64(i+1) * step)}
		}
		return out
	}

	spaceUnits := 1.0
	totalUnits := 0.0
	wordUnits := make([]float64, len(words))
	for i, w := range words {
		u := float64(maxInt(utf8.RuneCountInString(w), 1))
		wordUnits[i] = u
		totalUnits += u
	}
	totalUnits += float64(maxInt(len(words)-1, 0)) * spaceUnits
	if totalUnits <= 0 {
		totalUnits = float64(len(words))
	}

	usable := x2 - x1
	unitW := usable / totalUnits
	spaceW := unitW * spaceUnits
	out := make([]syntheticWordSpan, 0, len(words))
	cursor := x1
	for i, u := range wordUnits {
		ww := maxFloat(0.9, u*unitW)
		wx1 := cursor
		wx2 := minFloat(wx1+ww, x2)
		if i == len(wordUnits)-1 {
			wx2 = x2
		}
		out = append(out, syntheticWordSpan{X1: wx1, X2: wx2})
		cursor = minFloat(wx2+spaceW, x2)
	}
	return out
}

func normalizeSyntheticWords(words []string) []string {
	if len(words) == 0 {
		return words
	}

	out := make([]string, 0, len(words))
	prefix := ""
	for _, raw := range words {
		word := strings.TrimSpace(raw)
		if word == "" {
			continue
		}

		if isStandaloneSyntheticPunct(word) {
			if shouldAttachPunctRight(word) {
				prefix += word
				continue
			}
			if len(out) > 0 {
				out[len(out)-1] += word
			} else {
				prefix += word
			}
			continue
		}

		out = append(out, prefix+word)
		prefix = ""
	}

	if prefix != "" {
		if len(out) > 0 {
			out[len(out)-1] += prefix
		} else {
			out = append(out, prefix)
		}
	}

	if len(out) == 0 {
		return []string{" "}
	}
	return out
}

func isStandaloneSyntheticPunct(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return false
		}
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return true
}

func shouldAttachPunctRight(s string) bool {
	return s == "/" || s == "(" || s == "[" || s == "{"
}

func detectTextRegion(img image.Image) syntheticLayoutRegion {
	if img == nil {
		return syntheticLayoutRegion{}
	}

	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if w <= 0 || h <= 0 {
		return syntheticLayoutRegion{}
	}

	step := maxInt(1, minInt(w, h)/1200)
	// Luma threshold around 240/255 in 16-bit range.
	darkThreshold := uint32(240 * 257)

	colsSampled := (w + step - 1) / step
	rowsSampled := (h + step - 1) / step
	colCounts := make([]int, colsSampled)
	rowCounts := make([]int, rowsSampled)

	minX := w
	minY := h
	maxX := -1
	maxY := -1
	darkCount := 0
	sampled := 0

	for y := b.Min.Y; y < b.Max.Y; y += step {
		for x := b.Min.X; x < b.Max.X; x += step {
			r, g, bl, a := img.At(x, y).RGBA()
			if a == 0 {
				continue
			}
			sampled++
			// Integer luma approximation.
			luma := (299*r + 587*g + 114*bl) / 1000
			if luma < darkThreshold {
				darkCount++
				lx := x - b.Min.X
				ly := y - b.Min.Y
				cx := lx / step
				ry := ly / step
				if cx >= 0 && cx < len(colCounts) {
					colCounts[cx]++
				}
				if ry >= 0 && ry < len(rowCounts) {
					rowCounts[ry]++
				}
				if lx < minX {
					minX = lx
				}
				if ly < minY {
					minY = ly
				}
				if lx > maxX {
					maxX = lx
				}
				if ly > maxY {
					maxY = ly
				}
			}
		}
	}

	if sampled == 0 || darkCount < 50 {
		return syntheticLayoutRegion{}
	}
	coverage := float64(darkCount) / float64(sampled)
	if coverage < 0.0005 {
		return syntheticLayoutRegion{}
	}

	rowThreshold := maxInt(2, colsSampled/220) // ~0.45% of sampled columns
	colThreshold := maxInt(2, rowsSampled/220) // ~0.45% of sampled rows
	rowStart, rowEnd, rowOK := largestDenseSpan(rowCounts, rowThreshold)
	colStart, colEnd, colOK := largestDenseSpan(colCounts, colThreshold)
	if !rowOK || !colOK {
		return syntheticLayoutRegion{}
	}
	// Expand horizontal coverage using weighted dark-pixel quantiles.
	// Dense-span alone can under-estimate sparse right-side text columns on receipts.
	if qStart, qEnd, qOK := weightedSpan(colCounts, 0.001, 0.999); qOK {
		if qStart < colStart {
			colStart = qStart
		}
		if qEnd > colEnd {
			colEnd = qEnd
		}
	}

	minY = rowStart * step
	maxY = minInt(h-1, ((rowEnd+1)*step)-1)
	minX = colStart * step
	maxX = minInt(w-1, ((colEnd+1)*step)-1)

	if maxX <= minX || maxY <= minY {
		return syntheticLayoutRegion{}
	}

	pad := maxInt(3, step*4)
	minX = maxInt(0, minX-pad)
	minY = maxInt(0, minY-pad)
	maxX = minInt(w-1, maxX+pad)
	maxY = minInt(h-1, maxY+pad)

	return syntheticLayoutRegion{
		X1:    float64(minX),
		Y1:    float64(minY),
		X2:    float64(maxX),
		Y2:    float64(maxY),
		Valid: true,
	}
}

func detectTextLineBands(img image.Image, region syntheticLayoutRegion) []syntheticLineBand {
	if img == nil {
		return nil
	}
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}

	step := maxInt(1, minInt(w, h)/1500)
	darkThreshold := uint32(240 * 257)

	xMin := 0
	xMax := w - 1
	if region.Valid {
		padX := maxInt(3, step*8)
		xMin = clampInt(int(region.X1)-padX, 0, w-1)
		xMax = clampInt(int(region.X2)+padX, xMin+1, w-1)
	}
	if xMax <= xMin {
		return nil
	}

	rowsSampled := (h + step - 1) / step
	colsSampled := ((xMax - xMin + 1) + step - 1) / step
	if rowsSampled <= 0 || colsSampled <= 0 {
		return nil
	}
	rowDark := make([]int, rowsSampled)
	rowMinX := make([]int, rowsSampled)
	rowMaxX := make([]int, rowsSampled)
	for i := range rowMinX {
		rowMinX[i] = w
		rowMaxX[i] = -1
	}

	for y := 0; y < h; y += step {
		ry := y / step
		if ry < 0 || ry >= rowsSampled {
			continue
		}
		for x := xMin; x <= xMax; x += step {
			r, g, bl, a := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
			if a == 0 {
				continue
			}
			luma := (299*r + 587*g + 114*bl) / 1000
			if luma < darkThreshold {
				rowDark[ry]++
				if x < rowMinX[ry] {
					rowMinX[ry] = x
				}
				if x > rowMaxX[ry] {
					rowMaxX[ry] = x
				}
			}
		}
	}

	rowThreshold := maxInt(2, colsSampled/90)
	active := make([]bool, rowsSampled)
	activeCount := 0
	for i, c := range rowDark {
		if c >= rowThreshold {
			active[i] = true
			activeCount++
		}
	}
	if activeCount < 2 {
		return nil
	}

	type run struct{ s, e int }
	runs := make([]run, 0, activeCount/2)
	start := -1
	gap := 0
	maxGap := 1
	for i := 0; i < rowsSampled; i++ {
		if active[i] {
			if start < 0 {
				start = i
			}
			gap = 0
			continue
		}
		if start >= 0 {
			gap++
			if gap > maxGap {
				end := i - gap
				if end >= start {
					runs = append(runs, run{s: start, e: end})
				}
				start = -1
				gap = 0
			}
		}
	}
	if start >= 0 {
		runs = append(runs, run{s: start, e: rowsSampled - 1})
	}
	if len(runs) == 0 {
		return nil
	}

	out := make([]syntheticLineBand, 0, len(runs))
	for _, r := range runs {
		minXR := w
		maxXR := -1
		for i := r.s; i <= r.e && i < rowsSampled; i++ {
			if rowDark[i] == 0 {
				continue
			}
			if rowMinX[i] < minXR {
				minXR = rowMinX[i]
			}
			if rowMaxX[i] > maxXR {
				maxXR = rowMaxX[i]
			}
		}
		if maxXR <= minXR {
			continue
		}
		y1 := clampFloat(float64(r.s*step), 0, float64(h-1))
		y2 := clampFloat(float64((r.e+1)*step-1), y1+1, float64(h-1))
		bandYMin := maxInt(0, r.s*step)
		bandYMax := minInt(h-1, ((r.e+1)*step)-1)
		wordSpans := detectWordSpansInBand(img, minXR, maxXR, bandYMin, bandYMax, step, darkThreshold)
		bandPad := maxInt(2, step*2)
		out = append(out, syntheticLineBand{
			X1:        float64(maxInt(0, minXR-bandPad)),
			Y1:        y1,
			X2:        float64(minInt(w-1, maxXR+bandPad)),
			Y2:        y2,
			WordSpans: wordSpans,
		})
	}
	return out
}

func detectWordSpansInBand(
	img image.Image,
	xMin, xMax, yMin, yMax, step int,
	darkThreshold uint32,
) []syntheticWordSpan {
	if img == nil || xMax <= xMin || yMax <= yMin {
		return nil
	}
	b := img.Bounds()
	w := b.Dx()
	h := b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}

	xMin = clampInt(xMin, 0, w-1)
	xMax = clampInt(xMax, xMin+1, w-1)
	yMin = clampInt(yMin, 0, h-1)
	yMax = clampInt(yMax, yMin+1, h-1)
	step = maxInt(1, step)

	cols := ((xMax - xMin + 1) + step - 1) / step
	rows := ((yMax - yMin + 1) + step - 1) / step
	if cols <= 0 || rows <= 0 {
		return nil
	}

	colDark := make([]int, cols)
	for x := xMin; x <= xMax; x += step {
		cx := (x - xMin) / step
		if cx < 0 || cx >= cols {
			continue
		}
		dark := 0
		for y := yMin; y <= yMax; y += step {
			r, g, bl, a := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
			if a == 0 {
				continue
			}
			luma := (299*r + 587*g + 114*bl) / 1000
			if luma < darkThreshold {
				dark++
			}
		}
		colDark[cx] = dark
	}

	colThreshold := maxInt(1, rows/8)
	active := make([]bool, cols)
	activeCount := 0
	for i, c := range colDark {
		if c >= colThreshold {
			active[i] = true
			activeCount++
		}
	}
	if activeCount == 0 {
		return nil
	}

	type run struct{ s, e int }
	runs := make([]run, 0, activeCount/2)
	start := -1
	gap := 0
	maxGap := 1
	for i := 0; i < cols; i++ {
		if active[i] {
			if start < 0 {
				start = i
			}
			gap = 0
			continue
		}
		if start >= 0 {
			gap++
			if gap > maxGap {
				end := i - gap
				if end >= start {
					runs = append(runs, run{s: start, e: end})
				}
				start = -1
				gap = 0
			}
		}
	}
	if start >= 0 {
		runs = append(runs, run{s: start, e: cols - 1})
	}
	if len(runs) == 0 {
		return nil
	}

	spans := make([]syntheticWordSpan, 0, len(runs))
	for _, r := range runs {
		spanPad := maxInt(1, step/2)
		wx1 := float64(clampInt(xMin+(r.s*step)-spanPad, 0, w-2))
		wx2 := float64(clampInt(xMin+((r.e+1)*step)+spanPad, 1, w-1))
		if wx2-wx1 < maxFloat(1.0, float64(step)*1.6) {
			continue
		}
		spans = append(spans, syntheticWordSpan{X1: wx1, X2: wx2})
	}
	if len(spans) <= 1 {
		return spans
	}

	// Merge tiny gaps that usually split the same visual word.
	mergeGap := maxFloat(0.8, float64(step)*1.2)
	merged := make([]syntheticWordSpan, 0, len(spans))
	cur := spans[0]
	for i := 1; i < len(spans); i++ {
		if spans[i].X1-cur.X2 <= mergeGap {
			cur.X2 = maxFloat(cur.X2, spans[i].X2)
			continue
		}
		merged = append(merged, cur)
		cur = spans[i]
	}
	merged = append(merged, cur)
	return merged
}

func largestDenseSpan(counts []int, threshold int) (int, int, bool) {
	if len(counts) == 0 {
		return 0, 0, false
	}
	if threshold < 1 {
		threshold = 1
	}

	bestStart := -1
	bestEnd := -1
	bestScore := -1

	curStart := -1
	curScore := 0
	for i, c := range counts {
		if c >= threshold {
			if curStart < 0 {
				curStart = i
				curScore = 0
			}
			curScore += c
			continue
		}
		if curStart >= 0 {
			curEnd := i - 1
			spanLen := curEnd - curStart + 1
			score := curScore + spanLen*threshold
			if score > bestScore {
				bestStart = curStart
				bestEnd = curEnd
				bestScore = score
			}
			curStart = -1
			curScore = 0
		}
	}
	if curStart >= 0 {
		curEnd := len(counts) - 1
		spanLen := curEnd - curStart + 1
		score := curScore + spanLen*threshold
		if score > bestScore {
			bestStart = curStart
			bestEnd = curEnd
			bestScore = score
		}
	}
	if bestStart < 0 || bestEnd < bestStart {
		return 0, 0, false
	}
	return bestStart, bestEnd, true
}

func weightedSpan(counts []int, lowQ, highQ float64) (int, int, bool) {
	if len(counts) == 0 {
		return 0, 0, false
	}
	if lowQ < 0 {
		lowQ = 0
	}
	if highQ > 1 {
		highQ = 1
	}
	if highQ <= lowQ {
		highQ = minFloat(lowQ+0.01, 1)
	}

	total := 0
	for _, c := range counts {
		if c > 0 {
			total += c
		}
	}
	if total <= 0 {
		return 0, 0, false
	}

	lowTarget := int(float64(total) * lowQ)
	highTarget := int(float64(total) * highQ)
	cum := 0
	start := 0
	end := len(counts) - 1
	foundStart := false
	for i, c := range counts {
		if c < 0 {
			c = 0
		}
		cum += c
		if !foundStart && cum >= lowTarget {
			start = i
			foundStart = true
		}
		if cum >= highTarget {
			end = i
			break
		}
	}
	if end < start {
		end = start
	}
	return start, end, true
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func clampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func normalizeSyntheticPageSize(w, h float64) (float64, float64) {
	// Incoming image dimensions are often in high-resolution pixels (e.g. 5000x7000).
	// Keep aspect ratio but normalize to PDF-point space to avoid oversized pages
	// that can break text-layer selection in some viewers.
	longEdge := maxFloat(w, h)
	if longEdge <= 2000 {
		return w, h
	}

	// Target a typical PDF page long edge (~A4 in points) for better viewer compatibility.
	scale := 842.0 / longEdge
	nw := maxFloat(200, w*scale)
	nh := maxFloat(200, h*scale)
	return nw, nh
}

// createOpenAIClient creates a new OpenAI vision model client
func createOpenAIClient(config Config) (llms.Model, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is not set")
	}
	opts := []openai.Option{
		openai.WithModel(config.VisionLLMModel),
		openai.WithToken(apiKey),
		openai.WithHTTPClient(createInstrumentedHTTPClient()),
	}
	// In our deployment this points to the local OCR router: http://glm-ocr-router:8088/v1
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		opts = append(opts, openai.WithBaseURL(baseURL))
	}
	return openai.New(opts...)
}

// createOllamaClient creates a new Ollama vision model client
func createOllamaClient(config Config) (llms.Model, error) {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	opts := []ollama.Option{
		ollama.WithModel(config.VisionLLMModel),
		ollama.WithServerURL(host),
	}
	if config.OllamaContextLength > 0 {
		opts = append(opts, ollama.WithRunnerNumCtx(config.OllamaContextLength))
	}
	return ollama.New(opts...)
}

// createMistralClient creates a new Mistral vision model client
func createMistralClient(config Config) (llms.Model, error) {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("Mistral API key is not set")
	}
	return mistral.New(
		mistral.WithModel(config.VisionLLMModel),
		mistral.WithAPIKey(apiKey),
	)
}

// createAnthropicClient creates a new Anthropic vision model client
func createAnthropicClient(config Config) (llms.Model, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("Anthropic API key is not set")
	}
	return anthropic.New(
		anthropic.WithModel(config.VisionLLMModel),
		anthropic.WithToken(apiKey),
	)
}
