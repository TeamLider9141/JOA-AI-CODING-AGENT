import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    path: str          # relative to indexed repo root
    symbol: str        # e.g. "UserService.login" or "lines-1" for text
    kind: str           # function | method | class | text
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    text: str

    @property
    def chunk_id(self) -> str:
        raw = f"{self.path}:{self.symbol}:{self.text}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))

    def payload(self) -> dict:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }
