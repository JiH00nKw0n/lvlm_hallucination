from urllib.parse import urlparse


def is_url(url_or_filename: str) -> bool:
    """
    Checks if a given string is a valid URL.

    Args:
        url_or_filename (str): A string that may represent a URL.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")