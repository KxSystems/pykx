from code import InteractiveConsole # noqa F401: used by `.pykx.console[]`
import sys

from pykx.exceptions import QError

from . import _pykx_helpers


class QConsole:
    """Emulated q console that can be dropped into from the Python console."""
    def __init__(self, q):
        self._q = q
        self.k_mode = False

    def _get_prompt(self) -> str:
        if self.k_mode:
            return '  '
        ctx = str(self._q._call('system"d"', wait=True))
        return f'q{ctx if ctx != "." else ""})'

    def _eval_and_print(self, code: str) -> None:
        try:
            x = self._q._call('.Q.s value@', code.encode(), wait=True)
        except QError as ex:
            print(f"'{ex}", file=sys.stderr)
        else:
            print(x, end='')

    def __call__(self):
        """Activates the q console.

        Inputs will be sent to this basic/limited emulation of the q console until `ctrl-d` or
        two backslashes are input. A single backslash can be used to switch into k mode.

        Debugging mode is not supported.
        """
        while True:
            prompt = self._get_prompt()
            try:
                code = input(prompt)
            except EOFError:
                print()
                break
            stripped_code = code.strip()
            if stripped_code == r'\\':
                break
            elif stripped_code == '\\':
                self.k_mode = not self.k_mode
                continue
            elif stripped_code == '':
                continue
            if self.k_mode:
                code = 'k)' + code
            self._eval_and_print(code)


class PyConsole:
    def __init__(self):
        self.console = InteractiveConsole(globals())
        self.console.push('import sys')
        self.console.push('quit = sys.exit')
        self.console.push('exit = sys.exit')

    def interact(self, banner=None, exitmsg=None):
        try:
            self.console.interact(banner=banner, exitmsg=exitmsg)
        except SystemExit:
            _pykx_helpers.clean_errors()
