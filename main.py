import argparse
from pprint import pprint
from config import setup_environment
from graph import get_app

def print_stream(app, inputs):
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")
    
    # Process final generation based on the output keys
    docs = value.get('documents', [])
    if isinstance(docs, list) and len(docs) > 0:
        pprint(docs[0].metadata.get('description', docs[0].page_content))
    else:
        pprint(docs)

def main():
    parser = argparse.ArgumentParser(description="LangGraph AstraDB Agent")
    parser.add_argument(
        "question", 
        type=str, 
        help="The question you want to ask the agent."
    )
    args = parser.parse_args()
    
    # Initialize the basic environment configurations
    setup_environment()
    
    # Build the graph application
    app = get_app()
    
    inputs = {
        "question": args.question
    }
    print(f"Asking: {args.question}")
    print_stream(app, inputs)

if __name__ == "__main__":
    main()
