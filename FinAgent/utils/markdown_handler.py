def escape_dollars(content: str) -> str:
    """
    Ensure all dollar signs are escaped in the content.

    Args:
        content (str): LLM Response, which is sometimes incorrectly formatted 
            markdown, due to the common use of dollar signs in finance.

    Returns:
        str: Correctly escaped string.
    """
    return content.replace("\\$", "$").replace("$", "\\$")