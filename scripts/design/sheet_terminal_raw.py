import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import TerminalTheme
from gen import STYLES
from braille import braille
c = Console(record=True, width=150, force_terminal=True, color_system="truecolor")
for cols in (30, 16):
    t = Table.grid(padding=(0, 4))
    for _ in STYLES: t.add_column()
    t.add_row(*[braille(f, cols) for f in STYLES.values()])
    t.add_row(*[name for name in STYLES])
    c.print(f"[#8a90a0]terminal size: {cols} cols x {cols//2} rows[/]"); c.print(t)
c.save_svg(sys.argv[1], title="mitsudomoe in the terminal", theme=TerminalTheme((14,15,19),(225,228,235),[(0,0,0)]*8,[(0,0,0)]*8))
