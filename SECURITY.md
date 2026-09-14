# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities in public GitHub issues.

Email security@your-org.example with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

We will respond within 72 hours and work with you to address the issue before
public disclosure. We appreciate responsible disclosure.

## Security Considerations

- AgentBench never hard-codes API keys or credentials
- All secrets are loaded from environment variables
- The `.gitignore` prevents committing `.env` files
- HTTP adapter requests should use HTTPS in production
- The SQLite database stores evaluation results locally — treat it as you would any application data
