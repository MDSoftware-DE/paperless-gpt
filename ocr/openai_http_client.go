package ocr

import (
	"net/http"
	"strconv"
)

// ocrHeaderTransport injects request metadata into outgoing OpenAI HTTP calls.
//
// This is intentionally implemented at the HTTP transport layer so we can set
// per-request headers without changing langchaingo's OpenAI client internals.
type ocrHeaderTransport struct {
	base http.RoundTripper
}

func (t *ocrHeaderTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	if req != nil {
		meta := RequestMetaFromContext(req.Context())

		// Do not overwrite headers if already set by caller.
		if req.Header.Get("X-Title") == "" {
			req.Header.Set("X-Title", "paperless-gpt")
		}
		if req.Header.Get("X-OCR-Document-ID") == "" && meta.DocumentID != 0 {
			req.Header.Set("X-OCR-Document-ID", strconv.Itoa(meta.DocumentID))
		}
		if req.Header.Get("X-OCR-Original-Filename") == "" && meta.OriginalFileName != "" {
			req.Header.Set("X-OCR-Original-Filename", meta.OriginalFileName)
		}
		if req.Header.Get("X-OCR-Page") == "" && meta.PageNumber != 0 {
			req.Header.Set("X-OCR-Page", strconv.Itoa(meta.PageNumber))
		}
	}

	return base.RoundTrip(req)
}

func createInstrumentedHTTPClient() *http.Client {
	return &http.Client{
		Transport: &ocrHeaderTransport{base: http.DefaultTransport},
	}
}

