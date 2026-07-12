---
inclusion: fileMatch
fileMatchPattern: "src/app/shared/langchain_layer/**/*"
---

# LangChain Integration Patterns

## Best Practices

- Use conversation memory for chat interactions
- Implement proper token counting and limits
- Handle API rate limits gracefully
- Use callbacks for monitoring and logging
- Implement proper error handling for AI operations
- Cache frequently used prompts and chains
- Monitor costs and usage metrics
- Use streaming for long responses when possible

## Memory Management

Keep memory usage optimized:
- Use conversation memory for chat
- Implement token limits
- Clear old conversations periodically
- Monitor memory growth

## Rate Limiting

Handle API rate limits gracefully:
- Implement exponential backoff
- Track rate limit headers
- Queue requests when needed
- Log rate limit events

## Callbacks

Use callbacks for observability:
- Monitor token usage
- Track costs
- Log chain execution
- Measure latency

## Streaming

Use streaming for long responses:
- Improves perceived latency
- Reduces memory usage
- Better user experience

## Error Handling

Implement proper error handling:
- Catch API errors
- Implement retries
- Log failures
- Provide fallbacks
