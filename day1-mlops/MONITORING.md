# W8D1 ML API Monitoring Strategy

## 1. Metrics to Track

### Availability
- API uptime
- Health-check success rate
- Container restart count

### Performance
- Request latency
- Average response time
- P95 and P99 latency

### Traffic
- Requests per minute
- Requests per endpoint
- HTTP status-code distribution

### Model/API Quality
- Prediction distribution
- Confidence scores
- Error rate
- Invalid input rate

### Infrastructure
- CPU usage
- Memory usage
- Docker container health
- Disk usage

## 2. Alerts

### Critical Alerts
- API health check fails continuously
- HTTP 5xx error rate exceeds 5%
- Container repeatedly restarts

### Performance Alerts
- P95 latency exceeds 2 seconds
- Request timeout rate exceeds 2%

### Infrastructure Alerts
- Memory usage exceeds 80%
- CPU usage exceeds 85%

## 3. Retraining Triggers

A model retraining process should be considered when:

1. Prediction accuracy drops below the agreed threshold.
2. Input data distribution changes significantly.
3. Data drift is detected.
4. New labelled production data becomes available.
5. Business performance metrics decline consistently.

## 4. Monitoring Workflow

Production requests
        |
        v
ML API
        |
        +----> Logs
        |
        +----> Latency Metrics
        |
        +----> Error Metrics
        |
        +----> Prediction Metrics
        |
        v
Monitoring System
        |
        v
Alerts
        |
        v
Investigation
        |
        v
Retraining when required