FROM golang:1.24-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o /llm-router ./cmd/llm-router/

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /llm-router /usr/local/bin/llm-router
ENV PORT=11435
EXPOSE 11435
ENTRYPOINT ["llm-router"]
