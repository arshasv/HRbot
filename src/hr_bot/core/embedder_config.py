import os

def get_azure_openai_embedder():
    """Reusable embedder configuration for all crews."""
    return {
        "provider": "openai",
        "config": {
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_base": "https://genesisforge1.openai.azure.com/",
            "api_type": "azure",
            "api_version": "2023-05-15",
            "deployment_id": "text-embedding-3-small",
            "model": "text-embedding-3-small",
        },
    }
