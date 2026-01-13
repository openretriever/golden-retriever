
def main():
    print("Checking BEHAVIOR-1K environment...")
    
    try:
        import omnigibson
        print(f"✅ omnigibson imported successfully (version: {omnigibson.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import omnigibson: {e}")
        
    try:
        import behavior1k
        print(f"✅ behavior1k imported successfully from {behavior1k.__path__}")
    except ImportError as e:
        print(f"❌ Failed to import behavior1k: {e}")
        
    print("\nNote: This environment requires a Linux system with NVIDIA GPU for full simulation.")

if __name__ == "__main__":
    main()
