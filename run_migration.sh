#!/bin/bash
# Migration script for Retriever-Examples

set -e  # Exit on any error

echo "🚀 Starting Retriever-Examples Migration"
echo "========================================"

# Create utils directory if it doesn't exist
mkdir -p utils

# Make the migration script executable
chmod +x utils/migrate_components.py

# Run the migration
echo "📦 Running migration script..."
python utils/migrate_components.py

echo ""
echo "✅ Migration completed successfully!"
echo ""
echo "Next steps:"
echo "1. cd pi0"
echo "2. pip install -e ."
echo "3. Test with: python examples/integration_demo.py"
