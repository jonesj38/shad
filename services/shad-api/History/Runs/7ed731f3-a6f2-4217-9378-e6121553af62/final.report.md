# Run Report: 7ed731f3-a6f2-4217-9378-e6121553af62

**Goal:** Create a simple Python function that validates email addresses

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 220

## Result

Here's a simple Python function to validate email addresses:

```python
import re

def validate_email(email: str) -> bool:
    """
    Validate an email address using a regex pattern.
    
    Args:
        email: The email address to validate.
    
    Returns:
        True if the email is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

**Usage:**

```python
print(validate_email("user@example.com"))      # True
print(validate_email("invalid-email"))          # False
print(validate_email("user.name+tag@domain.co.uk"))  # True
```

**What this validates:**
- Local part: letters, numbers, dots, underscores, percent, plus, hyphen
- Single `@` symbol
- Domain: letters, numbers, dots, hyphens
- TLD: at least 2 letters

For production use, consider the `email-validator` library (`pip install email-validator`) which handles edge cases and can verify DNS records.
