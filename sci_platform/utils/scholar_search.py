import logging
import requests
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


def return_empty_list_on_failure(retry_state):
    """Callback function to return empty list when all retries fail"""
    logger.error(f"All retries failed. Last exception: {retry_state.outcome.exception()}")
    return {"total": 0, "papers": []}

@retry(
    stop=stop_after_attempt(100),
    wait=wait_exponential(multiplier=5, min=1, max=5),  # Max wait time extended to 60 seconds
    retry=retry_if_exception_type((
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError
    )),
    retry_error_callback=return_empty_list_on_failure
)
def semantic_scholar_search(query: str, result_limit: int) -> Dict[str, Any]:
    """Helper method for making Semantic Scholar API requests
    
    Args:
        query: Search query string
        result_limit: Maximum number of results to return
        
    Returns:
        JSON response data from API
        
    Raises:
        requests.exceptions.HTTPError: When API returns non-200 status code
        requests.exceptions.RequestException: When request fails
    """

    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,abstract,publicationTypes",
        },
        timeout=5
    )
    
    if response.status_code == 429:
        logger.warning("API rate limit exceeded. Will retry with exponential backoff...")
    
    response.raise_for_status()
    return response.json()
