package ocr

import "context"

// RequestMeta is propagated via context into the OpenAI HTTP request so that
// intermediaries (like the OCR router) can persist artifacts with stable IDs.
type RequestMeta struct {
	DocumentID       int
	OriginalFileName string
	PageNumber       int
}

type requestMetaKey struct{}

var ctxRequestMetaKey = &requestMetaKey{}

// WithRequestMeta merges the provided meta into any existing meta on ctx.
// Zero values do not overwrite existing values.
func WithRequestMeta(ctx context.Context, add RequestMeta) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	cur := RequestMetaFromContext(ctx)

	if add.DocumentID != 0 {
		cur.DocumentID = add.DocumentID
	}
	if add.OriginalFileName != "" {
		cur.OriginalFileName = add.OriginalFileName
	}
	if add.PageNumber != 0 {
		cur.PageNumber = add.PageNumber
	}

	return context.WithValue(ctx, ctxRequestMetaKey, cur)
}

func RequestMetaFromContext(ctx context.Context) RequestMeta {
	if ctx == nil {
		return RequestMeta{}
	}
	v := ctx.Value(ctxRequestMetaKey)
	m, ok := v.(RequestMeta)
	if !ok {
		return RequestMeta{}
	}
	return m
}

