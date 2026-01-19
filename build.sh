#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Fractal Explorer...${NC}"

# Create build directory
mkdir -p build
cd build

# Run CMake
echo -e "${GREEN}Running CMake...${NC}"
cmake .. || { echo -e "${RED}CMake failed${NC}"; exit 1; }

# Build
echo -e "${GREEN}Compiling...${NC}"
ninja || make -j$(nproc) || { echo -e "${RED}Build failed${NC}"; exit 1; }

echo -e "${GREEN}Build successful!${NC}"
echo ""
echo "Run with: ./build/fractal-explorer"
echo "Or with custom resolution: ./build/fractal-explorer 1920 1080"
