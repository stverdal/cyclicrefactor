#!/usr/bin/env python3
"""Test script to verify RAG integration with the pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from orchestrator import Orchestrator

def main():
    # Load config  
    config = load_config()
    print(f"Config loaded: retriever.persist_dir={config.retriever.persist_dir}")

    # Create orchestrator
    print("\nCreating orchestrator...")
    orch = Orchestrator(config=config)

    # Check RAG service status
    print(f"RAG service available: {orch.rag_service is not None}")
    if orch.rag_service:
        print(f"RAG service can query: {orch.rag_service.is_available()}")

    # Create a minimal test cycle
    test_cycle = {
        'id': 'test-rag-001',
        'graph': {
            'nodes': ['ModuleA', 'ModuleB'],
            'edges': [['ModuleA', 'ModuleB'], ['ModuleB', 'ModuleA']]
        },
        'files': [{
            'path': 'test.cs',
            'content': 'public class Test { }'
        }]
    }

    print('\nRunning pipeline...')
    results = orch.run_pipeline(test_cycle)
    print(f"Pipeline status: {results.get('status')}")
    
    desc = results.get("description", {})
    if isinstance(desc, dict):
        text = desc.get("text", "")[:300]
    else:
        text = str(desc)[:300]
    print(f"Description preview: {text}...")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
