import os
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

__all__ = ["help"]
_filepath = os.path.join(
    os.path.dirname(__file__), '..', '..', 'docs', 'api',
    'pykx-execution', 'q.md'
)
_base_url = "https://code.kx.com/pykx/4.0/api/pykx-execution/q.html"


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


def _fetch_web_documentation(keyword):
    """Fetch documentation from the PyKX website."""
    url = _base_url
    try:
        req = Request(url, headers={'User-Agent': 'PyKX/help'})
        with urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')

            # Extract the article content
            article_match = re.search(
                r'<article[^>]*class="[^"]*md-content__inner[^"]*"[^>]*>'
                r'(.*?)</article>',
                html, re.DOTALL
            )
            if not article_match:
                return None

            article_html = article_match.group(1)

            # Find the h3 section for this keyword
            # Pattern: <h3 id="keyword">...</h3> ... content ... <h3 id="next">
            pattern = rf'<h3[^>]*id="{keyword}"[^>]*>.*?</h3>(.*?)(?=<h3[^>]*id="|$)'
            section_match = re.search(pattern, article_html, re.DOTALL)

            if not section_match:
                return None

            section_html = section_match.group(1)
            text = f" • {keyword}\n\n"

            # Extract paragraphs
            for para_match in re.finditer(r'<p(?:\s[^>]*)?>(.*?)</p>', section_html, re.DOTALL):
                para_text = re.sub(r'<[^>]+>', '', para_match.group(1))
                para_text = _decode_html_entities(para_text).strip()
                if para_text:
                    text += para_text + "\n\n"

            # Extract code blocks
            for code_match in re.finditer(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>',
                                          section_html, re.DOTALL):
                code_text = _decode_html_entities(code_match.group(1)).strip()
                if code_text:
                    text += "\n"
                    for line in code_text.split("\n"):
                        text += "    " + line + "\n"
                    text += "\n"

            return text

    except (URLError, HTTPError, TimeoutError):
        return None
    except Exception:
        return None


def _decode_html_entities(text):
    """Decode common HTML entities."""
    return (text.replace('&lt;', '<')
            .replace('&gt;', '>')
            .replace('&amp;', '&')
            .replace('&quot;', '"')
            .replace('&#39;', "'")
            .replace('&nbsp;', ' ')
            .replace('&para;', '')  # Remove paragraph symbols
            .replace('&middot;', '·'))


def _load_markdown_section(keyword, file_path=_filepath):
    """Load and parse markdown section for a keyword."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        in_section = False
        section_lines = []
        for line in lines:
            if line.startswith(f'### [{keyword}]'):
                in_section = True
                section_lines.append(line)
            elif line.startswith('###') and in_section:
                break
            elif in_section:
                section_lines.append(line)

        if not section_lines:
            return None

        return section_lines
    except IOError:
        return None


def qhelp(keyword):
    """Parse and format the documentation for a given keyword."""
    markdown_lines = _load_markdown_section(keyword)
    if markdown_lines:
        text = f" • {keyword}\n\n"

        in_code_block = False
        table_lines = []
        in_table = False

        for i, line in enumerate(markdown_lines):
            if i == 0:
                continue

            if line.strip().startswith('```'):
                if in_code_block:
                    in_code_block = False
                    text += "\n"
                else:
                    in_code_block = True
                    text += "\n"
                continue

            if in_code_block:
                text += "    " + line.rstrip() + "\n"
                continue

            if '|' in line and not line.strip().startswith('#'):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
                continue
            elif in_table:
                if table_lines:
                    text += _parse_markdown_table(table_lines)
                in_table = False
                table_lines = []

            if line.strip() == '':
                if text.endswith('\n\n'):
                    continue
                text += "\n"
                continue

            if line.strip().startswith('- ') or line.strip().startswith('* '):
                list_text = line.strip()[2:]
                list_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', list_text)
                text += "  - " + list_text + "\n"
                continue

            if not line.startswith('#'):
                line_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line.strip())
                line_text = re.sub(r'`([^`]+)`', r'\1', line_text)
                if line_text:
                    text += line_text + "\n"

        if in_table and table_lines:
            text += _parse_markdown_table(table_lines)

        return text

    # Fallback to web scraping if local file fails
    web_result = _fetch_web_documentation(keyword)
    if web_result:
        return web_result

    # If both local and web sources fail
    return (f" • {keyword}\n\n"
            f"Documentation not available. Please check your internet connection\n"
            f"or visit {_base_url}#{keyword} for documentation.")


def _parse_markdown_table(lines):
    """Parse markdown table into ASCII table format."""
    if not lines:
        return ""

    rows = []
    header = None

    for line in lines:
        if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
            continue

        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]

        if cells:
            if header is None:
                header = cells
            else:
                rows.append(cells)

    if header and rows:
        return __ascii_table(rows, border=True, header=header) + "\n"
    return ""


def __ascii_table(table: list, header: list = None, align='left', border=False):
    """Converts a list of lists into an ASCII table."""
    if header is None:
        header = []
    widths = []
    for i in range(max(map(len, table))):
        widths.append(max(max(map(len, [row[i] for row in table if len(row) > i])),
                          len(header[i]) if len(header) > i else 0))

    printable = []

    if border:
        print_row = []
        for i in range(max(map(len, table))):
            if i > 0 and i < max(map(len, table)) - 1:
                print_row.append('─' * (widths[i] + 2))
            else:
                print_row.append('─' * (widths[i] + 1))
        printable.append('┌─' + '┬'.join(print_row) + '─┐')

    # header formatting
    if len(header) > 0:
        print_row = []
        for i in range(len(header)):
            assert header[i]
            if align == 'center':
                print_row.append(header[i].center(widths[i]))
            elif align == 'left':
                print_row.append(header[i].ljust(widths[i]))
            elif align == 'right':
                print_row.append(header[i].rjust(widths[i]))

        if border:
            printable.append('│ ' + ' │ '.join(print_row) + ' │')
        else:
            printable.append(' │ '.join(print_row))

        print_row = []
        for i in range(len(header)):
            if i > 0 and i < len(header) - 1:
                print_row.append('─' * (widths[i] + 2))
            else:
                print_row.append('─' * (widths[i] + 1))

        if border:
            printable.append('├─' + '┼'.join(print_row) + '─┤')
        else:
            printable.append('┼'.join(print_row))

    # table formatting
    for row in table:
        print_row = []
        for _ in range(len(widths) - len(row)):
            row.append('')

        for i in range(len(row)):
            if align == 'center':
                print_row.append(row[i].center(widths[i]))
            elif align == 'left':
                print_row.append(row[i].ljust(widths[i]))
            elif align == 'right':
                print_row.append(row[i].rjust(widths[i]))

        if border:
            printable.append('│ ' + ' │ '.join(print_row) + ' │')
        else:
            printable.append(' │ '.join(print_row))

    if border:
        print_row = []
        for i in range(max(map(len, table))):
            if i > 0 and i < max(map(len, table)) - 1:
                print_row.append('─' * (widths[i] + 2))
            else:
                print_row.append('─' * (widths[i] + 1))
        printable.append('└─' + '┴'.join(print_row) + '─┘')

    result = '\n'.join(printable)
    return result
