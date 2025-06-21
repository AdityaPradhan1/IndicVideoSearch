import sys
import subprocess
import os

def setup_environment():
    """Setup environment before any other imports"""
    
    # Install pysqlite3-binary
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pysqlite3-binary", "--quiet", "--disable-pip-version-check"
        ])
        print("✓ pysqlite3-binary installed")
    except Exception as e:
        print(f"Warning: Could not install pysqlite3-binary: {e}")
    
    # Replace sqlite3 module BEFORE any other imports
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
        print(f"✓ SQLite replaced with version {pysqlite3.sqlite_version}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import pysqlite3: {e}")
        return False

if __name__ == "__main__":
    setup_environment()
