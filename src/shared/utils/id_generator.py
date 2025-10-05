"""ID generation utilities using nanoid."""

try:
    from nanoid import generate
    NANOID_AVAILABLE = True
except ImportError:
    import uuid
    NANOID_AVAILABLE = False
    
    def generate(size=21):
        """Fallback to UUID if nanoid is not available."""
        return str(uuid.uuid4())


def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracking.
    
    Returns:
        A short, URL-safe correlation ID (12 characters if nanoid is available)
    """
    return generate(size=12)


def generate_session_id() -> str:
    """Generate a session ID for user sessions.
    
    Returns:
        A medium-length, URL-safe session ID (16 characters if nanoid is available)
    """
    return generate(size=16)


def generate_job_id(prefix: str = "") -> str:
    """Generate a job ID for background tasks.
    
    Args:
        prefix: Optional prefix for the job ID
        
    Returns:
        A job ID with optional prefix (10 characters if nanoid is available)
    """
    job_id = generate(size=10)
    return f"{prefix}_{job_id}" if prefix else job_id


def generate_document_id(prefix: str = "doc") -> str:
    """Generate a document ID.
    
    Args:
        prefix: Prefix for the document ID (default: "doc")
        
    Returns:
        A document ID with prefix (14 characters if nanoid is available)
    """
    doc_id = generate(size=14)
    return f"{prefix}_{doc_id}"


def generate_short_id() -> str:
    """Generate a short ID for general use.
    
    Returns:
        A short, URL-safe ID (8 characters if nanoid is available)
    """
    return generate(size=8)


def generate_long_id() -> str:
    """Generate a long ID for cases requiring more uniqueness.
    
    Returns:
        A long, URL-safe ID (21 characters - nanoid default, or UUID if fallback)
    """
    return generate()


def is_nanoid_available() -> bool:
    """Check if nanoid is available.
    
    Returns:
        True if nanoid is installed and available, False otherwise
    """
    return NANOID_AVAILABLE


# Convenience aliases
correlation_id = generate_correlation_id
session_id = generate_session_id
job_id = generate_job_id
document_id = generate_document_id
short_id = generate_short_id
long_id = generate_long_id