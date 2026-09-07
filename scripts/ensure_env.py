"""Cria o .env a partir do .env.example se ainda não existir."""

import pathlib
import shutil

dst = pathlib.Path(".env")
src = pathlib.Path(".env.example")

if dst.exists():
    print(".env ja existe")
else:
    shutil.copy(src, dst)
    print("criado .env a partir do .env.example - coloque sua GOOGLE_API_KEY")
