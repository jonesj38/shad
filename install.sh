#!/usr/bin/env bash
#
# Shad Installation Script
# Installs Shad - Shannon's Daemon for long-context AI reasoning
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/jonesj38/shad/main/install.sh | bash
#
# Or clone and run:
#   git clone https://github.com/jonesj38/shad.git && cd shad && ./install.sh
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SHAD_HOME="${SHAD_HOME:-$HOME/.shad}"
SHAD_REPO="https://github.com/jonesj38/shad.git"
SHAD_BRANCH="main"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for required dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    # Python 3.11+
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        local py_major=$(echo "$py_version" | cut -d. -f1)
        local py_minor=$(echo "$py_version" | cut -d. -f2)
        if [[ "$py_major" -lt 3 ]] || [[ "$py_major" -eq 3 && "$py_minor" -lt 11 ]]; then
            log_error "Python 3.11+ required (found $py_version)"
            missing+=("python3.11+")
        else
            log_success "Python $py_version"
        fi
    else
        log_error "Python 3 not found"
        missing+=("python3")
    fi

    # Git
    if command -v git &> /dev/null; then
        log_success "Git $(git --version | cut -d' ' -f3)"
    else
        missing+=("git")
    fi

    # Docker
    if command -v docker &> /dev/null; then
        log_success "Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"
    else
        missing+=("docker")
    fi

    # Docker Compose
    if docker compose version &> /dev/null; then
        log_success "Docker Compose $(docker compose version --short)"
    else
        missing+=("docker-compose")
    fi

    # Claude CLI (optional but recommended)
    if command -v claude &> /dev/null; then
        log_success "Claude CLI found"
    else
        log_warn "Claude CLI not found - install from https://claude.ai/code"
    fi

    # bun or npm (for qmd installation)
    if command -v bun &> /dev/null; then
        log_success "bun $(bun --version)"
        HAS_BUN=true
    elif command -v npm &> /dev/null; then
        log_success "npm $(npm --version)"
        HAS_NPM=true
    else
        log_warn "Neither bun nor npm found - qmd (semantic search) won't be installed"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo ""
        echo "Please install the missing dependencies and run the installer again."
        exit 1
    fi

    echo ""
}

# Clone or update the repository
setup_repo() {
    log_info "Setting up Shad repository..."

    mkdir -p "$SHAD_HOME"

    if [[ -d "$SHAD_HOME/repo/.git" ]]; then
        log_info "Updating existing repository..."
        cd "$SHAD_HOME/repo"
        git fetch origin
        git reset --hard "origin/$SHAD_BRANCH"
    else
        # Check if we're running from within the repo
        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        if [[ -f "$script_dir/services/shad-api/pyproject.toml" ]]; then
            log_info "Using local repository..."
            mkdir -p "$SHAD_HOME/repo"
            cp -r "$script_dir"/* "$SHAD_HOME/repo/"
        else
            log_info "Cloning repository..."
            git clone --depth 1 -b "$SHAD_BRANCH" "$SHAD_REPO" "$SHAD_HOME/repo"
        fi
    fi

    log_success "Repository ready at $SHAD_HOME/repo"
    echo ""
}

# Create virtual environment and install
setup_python() {
    log_info "Setting up Python environment..."

    cd "$SHAD_HOME/repo/services/shad-api"

    if [[ ! -d "$SHAD_HOME/venv" ]]; then
        python3 -m venv "$SHAD_HOME/venv"
        log_success "Created virtual environment"
    fi

    source "$SHAD_HOME/venv/bin/activate"

    log_info "Installing dependencies (this may take a minute)..."
    pip install --quiet --upgrade pip
    pip install --quiet -e .

    log_success "Python environment ready"
    echo ""
}

# Install qmd for semantic search (optional)
setup_qmd() {
    log_info "Setting up qmd (semantic search)..."

    # Check if qmd is already installed
    if command -v qmd &> /dev/null; then
        log_success "qmd already installed ($(qmd --version 2>/dev/null || echo 'version unknown'))"
        return 0
    fi

    # Try to install qmd
    if [[ "${HAS_BUN:-false}" == "true" ]]; then
        log_info "Installing qmd via bun..."
        if bun install -g https://github.com/tobi/qmd 2>/dev/null; then
            log_success "qmd installed via bun"
        else
            log_warn "Failed to install qmd via bun - using filesystem search fallback"
        fi
    elif [[ "${HAS_NPM:-false}" == "true" ]]; then
        log_info "Installing qmd via npm..."
        if npm install -g https://github.com/tobi/qmd 2>/dev/null; then
            log_success "qmd installed via npm"
        else
            log_warn "Failed to install qmd via npm - using filesystem search fallback"
        fi
    else
        log_warn "Skipping qmd installation (no bun/npm) - using filesystem search fallback"
        echo "  To enable semantic search later, install bun and run:"
        echo "    bun install -g https://github.com/tobi/qmd"
    fi

    echo ""
}

# Create wrapper scripts
setup_scripts() {
    log_info "Creating command scripts..."

    mkdir -p "$SHAD_HOME/bin"

    # Main shad wrapper
    cat > "$SHAD_HOME/bin/shad" << 'WRAPPER'
#!/usr/bin/env bash
# Shad CLI wrapper
SHAD_HOME="${SHAD_HOME:-$HOME/.shad}"
source "$SHAD_HOME/venv/bin/activate"
exec python -m shad.cli.main "$@"
WRAPPER
    chmod +x "$SHAD_HOME/bin/shad"

    log_success "Created shad command"
    echo ""
}

# Configure shell
setup_shell() {
    log_info "Configuring shell..."

    local shell_rc=""
    local path_line='export PATH="$HOME/.shad/bin:$PATH"'
    local home_line='export SHAD_HOME="$HOME/.shad"'

    # Detect shell
    if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ -n "${BASH_VERSION:-}" ]] || [[ "$SHELL" == *"bash"* ]]; then
        shell_rc="$HOME/.bashrc"
    fi

    if [[ -n "$shell_rc" ]] && [[ -f "$shell_rc" ]]; then
        # Add PATH if not already present
        if ! grep -q 'shad/bin' "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# Shad - Shannon's Daemon" >> "$shell_rc"
            echo "$home_line" >> "$shell_rc"
            echo "$path_line" >> "$shell_rc"
            log_success "Added Shad to $shell_rc"
        else
            log_info "Shad already in $shell_rc"
        fi
    else
        log_warn "Could not detect shell config file"
        echo ""
        echo "Add these lines to your shell config manually:"
        echo "  $home_line"
        echo "  $path_line"
    fi

    echo ""
}

# Print final instructions
print_success() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Shad installed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "To get started:"
    echo ""
    echo "  1. Restart your terminal or run:"
    echo -e "     ${BLUE}source ~/.zshrc${NC}  (or ~/.bashrc)"
    echo ""
    echo "  2. Start the Shad server:"
    echo -e "     ${BLUE}shad server start${NC}"
    echo ""
    echo "  3. Run your first task:"
    echo -e "     ${BLUE}shad run \"Hello, Shad!\"${NC}"
    echo ""
    echo "  4. Or with vault context:"
    echo -e "     ${BLUE}shad run \"Summarize my notes\" --vault ~/MyVault${NC}"
    echo ""
    echo "Commands:"
    echo "  shad server start   - Start Redis + API server"
    echo "  shad server stop    - Stop all services"
    echo "  shad server status  - Check service status"
    echo "  shad run            - Execute a reasoning task"
    echo "  shad status <id>    - Check run status"
    echo "  shad --help         - Show all commands"
    echo ""
    echo "Semantic Search (optional):"
    echo "  If qmd is installed, register your vault for hybrid search:"
    echo -e "     ${BLUE}qmd collection add ~/MyVault --name myvault${NC}"
    echo -e "     ${BLUE}qmd embed${NC}  # Generate embeddings"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Shad Installation${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    check_dependencies
    setup_repo
    setup_python
    setup_qmd
    setup_scripts
    setup_shell
    print_success
}

main "$@"
