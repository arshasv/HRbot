import os
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

def get_hr_policy_knowledge():
    """Initialize and return the HR policy knowledge source."""
    # Initialize PDF knowledge source
    KNOWLEDGE_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge")

    # Build correct absolute PDF path
    knowledge_dir = KNOWLEDGE_DIRECTORY
    os.makedirs(knowledge_dir, exist_ok=True)  # Create directory if it doesn't exist

    # ---------------------------------------------------------
    # Initialize PDF Knowledge Source
    # ---------------------------------------------------------
    # Get the directory where this script resides

    return PDFKnowledgeSource(
        file_paths=["HR_POLICY.pdf"],
        chunk_size=500,
        chunk_overlap=50,
        collection_name="hr_policy_collection",
        metadata={},
        safe_file_paths=[]
    )

