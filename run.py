"""Goldmap launcher — starts all components."""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_api():
    """Start the FastAPI backend."""
    from api.main import app
    import uvicorn
    
    from config.loader import CONFIG
    
    uvicorn.run(
        "api.main:app",
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        reload=True,
    )


def run_dashboard():
    """Start the Dash frontend."""
    from frontend.app import app
    from config.loader import CONFIG
    
    app.run_server(
        host=CONFIG["dashboard"]["host"],
        port=CONFIG["dashboard"]["port"],
        debug=True,
    )


def run_collector():
    """Start the data collector."""
    from core.collector import main
    asyncio.run(main())


def run_tests():
    """Run test suite."""
    import pytest
    pytest.main(["tests/", "-v", "--tb=short"])


def main():
    parser = argparse.ArgumentParser(
        description="Goldmap — Gold Market Intelligence Platform"
    )
    parser.add_argument(
        "component",
        choices=["api", "dashboard", "collector", "tests", "all"],
        help="Component to run",
    )
    
    args = parser.parse_args()
    
    if args.component == "api":
        run_api()
    elif args.component == "dashboard":
        run_dashboard()
    elif args.component == "collector":
        run_collector()
    elif args.component == "tests":
        run_tests()
    elif args.component == "all":
        print("Starting all components...")
        print("Use separate terminals for each:")
        print("  python run.py api        — FastAPI backend (port 8000)")
        print("  python run.py dashboard  — Dash frontend (port 8050)")
        print("  python run.py collector  — Data collector (background)")
        print("")
        print("Starting API + Dashboard together...")
        
        # Start API in background
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Run dashboard in foreground
        run_dashboard()


if __name__ == "__main__":
    main()