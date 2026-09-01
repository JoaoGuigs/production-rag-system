"""Inicia o servidor web."""

import uvicorn

from src.config import BASE_DIR


def main() -> None:
    uvicorn.run(
        "src.web.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BASE_DIR / "src"), str(BASE_DIR / "static")],
    )


if __name__ == "__main__":
    main()
