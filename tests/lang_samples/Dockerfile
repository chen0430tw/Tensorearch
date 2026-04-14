FROM golang:1.22-alpine AS builder
LABEL maintainer="test@example.com"
ARG VERSION=1.0.0
ENV CGO_ENABLED=0

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -ldflags "-s -w" -o /app/distrike .

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/distrike /usr/local/bin/
EXPOSE 8080
HEALTHCHECK --interval=30s CMD distrike status --json || exit 1
ENTRYPOINT ["distrike"]
CMD ["status"]
