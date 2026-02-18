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
	X1 float64
	Y1 float64
	X2 float64
	Y2 float64
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
		layoutRows += 1.0
		maxLineRunes = maxInt(maxLineRunes, utf8.RuneCountInString(line))
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

		// Only trust detected regions that are large enough to represent the real document text area.
		if regionWidth > w*0.15 && regionHeight > h*0.10 {
			// Expand slightly outward so selection starts near visual text edges.
			marginX = clampFloat(regionX1-maxFloat(2, w*0.004), 2, w-15)
			topMargin = clampFloat(regionY1-maxFloat(4, h*0.010), 2, h-15)

			usableWidth = clampFloat(
				regionWidth+maxFloat(4, w*0.008),
				20,
				maxFloat(20, w-marginX-5),
			)
			usableHeight = clampFloat(
				regionHeight+maxFloat(6, h*0.014),
				20,
				maxFloat(20, h-topMargin-5),
			)
			bottomMargin = maxFloat(5, h-(topMargin+usableHeight))
		}
	}
	lineAdvance := clampFloat(usableHeight/layoutRows, 6, 22)
	lineBoxHeight := clampFloat(lineAdvance*0.64, 4.2, lineAdvance*0.82)
	baseCharWidth := clampFloat(usableWidth/float64(maxLineRunes), 2.0, 5.4)
	baseSpaceWidth := maxFloat(baseCharWidth*0.88, 2.1)
	yCursor := topMargin

	scaledBands := make([]syntheticLineBand, 0, len(lineBands))
	for _, b := range lineBands {
		sb := syntheticLineBand{
			X1: clampFloat(b.X1*scaleX, 0, w-2),
			Y1: clampFloat(b.Y1*scaleY, 0, h-2),
			X2: clampFloat(b.X2*scaleX, 0, w-2),
			Y2: clampFloat(b.Y2*scaleY, 0, h-2),
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
	// Use detected line bands only when density is close to expected line count.
	// This avoids collapsing many OCR lines onto too few noisy bands.
	useBands := len(scaledBands) >= 8 && bandRatio >= 0.99 && bandRatio <= 1.03
	renderedLines := 0
	lastY2 := topMargin - lineBoxHeight
	// Start slightly above the estimated top margin so first visible lines
	// remain selectable even when region detection is conservative.
	startShift := minFloat(topMargin*0.92, lineAdvance*3.2)
	yCursor = maxFloat(2, topMargin-startShift)

	hocrLines := make([]hocr.Line, 0, len(lines))
	for i, lineText := range lines {
		if lineText == "" {
			yCursor += lineAdvance * 0.45
			continue
		}

		remainingLines := maxInt(nonEmptyLines-renderedLines, 1)
		availableHeight := maxFloat(8, (h-2)-yCursor)
		dynAdvance := minFloat(lineAdvance, availableHeight/float64(remainingLines))
		dynAdvance = clampFloat(dynAdvance, 3.8, lineAdvance)
		dynBoxHeight := clampFloat(lineBoxHeight*(dynAdvance/lineAdvance), 2.8, dynAdvance*0.82)

		fallbackY1 := yCursor
		y1 := fallbackY1
		if y1 >= h-5 {
			break
		}
		fallbackY2 := minFloat(y1+dynBoxHeight, h-2)
		y2 := fallbackY2
		if y2 <= y1 {
			y2 = minFloat(y1+12, h-2)
		}
		yCursor += dynAdvance

		words := normalizeSyntheticWords(strings.Fields(lineText))
		if len(words) == 0 {
			words = []string{" "}
		}

		lineMarginX := marginX
		lineUsableWidth := usableWidth
		if useBands {
			bi := mapLineIndex(renderedLines, nonEmptyLines, len(scaledBands))
			if bi >= 0 && bi < len(scaledBands) {
				band := scaledBands[bi]
					// Keep X placement stable from the synthetic model to avoid horizontal jitter.
					// Use detected bands only as a bounded vertical hint to keep selection smooth.
					fallbackCenter := (fallbackY1 + fallbackY2) / 2
					bandCenter := (band.Y1 + band.Y2) / 2
					maxShift := maxFloat(2, dynAdvance*0.30)
					centerShift := clampFloat(bandCenter-fallbackCenter, -maxShift, maxShift)
					targetCenter := fallbackCenter + centerShift
					targetHeight := clampFloat(band.Y2-band.Y1, dynBoxHeight*0.85, dynBoxHeight*1.05)
					y1 = targetCenter - (targetHeight / 2)
					y2 = targetCenter + (targetHeight / 2)
				}
			}
			// Enforce monotonic line progression; overlapping lines make UI text selection unusable.
			minY1 := lastY2 + maxFloat(1.6, dynAdvance*0.20)
			if y1 < minY1 {
				shift := minY1 - y1
				y1 += shift
				y2 += shift
		}
		if y2 > h-2 {
			shift := y2 - (h - 2)
			y1 -= shift
			y2 -= shift
		}
		if y1 < 2 {
			shift := 2 - y1
			y1 += shift
			y2 += shift
		}
		y1 = clampFloat(y1, 2, h-6)
		y2 = clampFloat(maxFloat(y2, y1+3), y1+3, h-2)
		lastY2 = y2

		lineRuneCount := 0
		for _, word := range words {
			lineRuneCount += maxInt(utf8.RuneCountInString(word), 1)
		}
		lineSpaceCount := maxInt(len(words)-1, 0)
		if lineRuneCount+lineSpaceCount <= 0 {
			lineRuneCount = 1
		}
			lineCharWidth := clampFloat(
				lineUsableWidth/float64(lineRuneCount+lineSpaceCount),
				baseCharWidth*0.76,
				baseCharWidth*1.12,
			)
			lineSpaceWidth := maxFloat(lineCharWidth*0.92, baseSpaceWidth)

		wordObjs := make([]hocr.Word, 0, len(words))
		xCursor := lineMarginX
		for wIdx, word := range words {
			charCount := maxInt(utf8.RuneCountInString(word), 1)
			ww := maxFloat(float64(charCount)*lineCharWidth, lineCharWidth)
			x1 := xCursor
			if x1 >= w-2 {
				break
			}
			x2 := minFloat(x1+ww, w-2)
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
			xCursor = minFloat(x2+lineSpaceWidth, w-2)
		}

		lineBBox := hocr.NewBoundingBox(lineMarginX, y1, minFloat(lineMarginX+lineUsableWidth, w-2), y2)
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

	minY = rowStart * step
	maxY = minInt(h-1, ((rowEnd+1)*step)-1)
	minX = colStart * step
	maxX = minInt(w-1, ((colEnd+1)*step)-1)

	if maxX <= minX || maxY <= minY {
		return syntheticLayoutRegion{}
	}

	pad := maxInt(2, step*2)
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
		xMin = clampInt(int(region.X1), 0, w-1)
		xMax = clampInt(int(region.X2), xMin+1, w-1)
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
		out = append(out, syntheticLineBand{
			X1: float64(maxInt(0, minXR-step)),
			Y1: y1,
			X2: float64(minInt(w-1, maxXR+step)),
			Y2: y2,
		})
	}
	return out
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
