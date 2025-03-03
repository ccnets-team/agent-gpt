import sys
import typer
import subprocess

def main():
    notice = typer.style(
        "Deprecation Notice: 'agent-gpt-aws' is deprecated. Please use 'agent-gpt' instead.\n"
        "Try running 'agent-gpt --help' for available commands.",
        fg=typer.colors.RED,
        bold=True
    )
    typer.echo(notice)
    
    # Use subprocess to call the 'agent-gpt --help' command and capture its output.
    try:
        result = subprocess.run(["agent-gpt", "--help"], capture_output=True, text=True, check=True)
        typer.echo(result.stdout, color=True)
    except Exception as e:
        typer.echo(f"Error retrieving help information: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
