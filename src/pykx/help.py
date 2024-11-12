import importlib.util
import warnings
import os

__all__ = ["help"]
_filepath = os.path.join(os.path.dirname(__file__), 'docs', 'api', 'pykx-execution', 'q.md')


def __dir__():
    return __all__


def _init(_q):
    global q
    q = _q


def _load_markdown_section(keyword, file_path=_filepath):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the specific section
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
        return "No description available for this function."

    # Convert the section to HTML
    section_markdown = ''.join(section_lines)

    md2_spec = importlib.util.find_spec("markdown2")
    if md2_spec is None:
        warnings.warn("The 'markdown2' package is required to use the 'qhelp' function.")
        return ""
    else:
        import markdown2
    section_html = markdown2.markdown(section_markdown, extras=["fenced-code-blocks"])

    return section_html


def qhelp(keyword):
    """Parse and format the documentation for a given keyword."""
    bs4_spec = importlib.util.find_spec("bs4")
    if bs4_spec is None:
        warnings.warn("The 'bs4' package is required to use the 'qhelp' function.")
        return ""
    else:
        from bs4 import BeautifulSoup

    html_content = _load_markdown_section(keyword)
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        tag = soup.find("h3", string=keyword)

        if tag is None:
            return None

        text = " • " + tag.text + "\n\n"
        for x in tag.findAllNext():
            if x.name == "h2":
                break

            elif x.name == "h3":
                break

            elif x.name == "p":
                only_link = True
                for w in x:
                    if w.name != "a":
                        only_link = False
                if not only_link:
                    text += x.text + "\n"

            elif x.name == "pre":
                text += "\n"
                for line in x.text.split("\n"):
                    text += "    " + line + "\n"

            elif x.name == "li":
                text += "  - " + x.text + "\n"

            elif x.name == "thead":
                table = []
                tmp = []
                for cell in x.text.split("\n"):
                    if cell != "":
                        tmp += [cell]
                table += [tmp]

            elif x.name == "tbody":
                for row in x.text.split("\n\n\n"):
                    tmp = []
                    for cell in row.split("\n"):
                        if cell != "":
                            tmp += [cell]
                    table += [tmp]
                text += __ascii_table(table[1:], border=True, header=table[0]) + "\n"

        return text
    else:
        print("html error")
        return None


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
