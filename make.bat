@echo off
if "%1"=="docker-dev" (
    echo Starting Docker development environment...
    docker-compose --profile dev up -d
    echo Services available:
    echo   Jupyter Lab: http://localhost:8888
    echo   Shell access: make docker-shell
) else if "%1"=="docker-test" (
    echo Running Docker tests...
    docker-compose run --rm test-unit
) else if "%1"=="docker-validate" (
    echo Running Docker validation...
    docker-compose run --rm validate
) else if "%1"=="help" (
    echo Available commands:
    echo   make.bat docker-dev      - Start development environment
    echo   make.bat docker-test     - Run tests
    echo   make.bat docker-validate - Quick validation
) else (
    echo Use: make.bat help
)